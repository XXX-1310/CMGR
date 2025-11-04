# here put the import lib
import pickle
import numpy as np
import torch
import torch.nn as nn
from models.BaseModel import BaseSeqModel
from models.SASRec import SASRecBackbone
from models.utils import Contrastive_Loss2, cal_bpr_loss
import torch.nn.functional as F
from geomloss import SamplesLoss

class MultiInterestExtractor(nn.Module):
    def __init__(self, args):
        super(MultiInterestExtractor, self).__init__()
        # cluster_emb = pickle.load(open("./data/{}/handled/cluster_emb.pkl".format(args.dataset), "rb"))
        # self.C = nn.Embedding.from_pretrained(torch.Tensor(cluster_emb), padding_idx=0, freeze=False)
        self.C = nn.Embedding(args.aspects, args.hidden_size)

    def forward(self, x_u):
        s_u = torch.matmul(self.C.weight, x_u.transpose(1, 2))
        s_u = s_u.permute(0, 2, 1)#B L C
        gumbel_weights = F.gumbel_softmax(s_u, tau=10, hard=False, dim=-1)  #B L C
        topk_weights, topk_indices = torch.topk(gumbel_weights, 10, dim=-1)  # B L K
        return topk_weights, topk_indices


class CMGR_base(BaseSeqModel):

    def __init__(self, user_num, item_num_dict, device, args) -> None:
        
        self.item_numA, self.item_numB = item_num_dict["0"], item_num_dict["1"]
        item_num =  self.item_numA + self.item_numB

        super().__init__(user_num, item_num, device, args)

        self.global_emb = args.global_emb

        # llm_emb_file = "item_emb"
        # llm_emb_file = "qwen_last"
        llm_emb_A = pickle.load(open("./data/{}/handled/{}_A_pca128.pkl".format(args.dataset, args.llm_emb_file), "rb"))
        llm_emb_B = pickle.load(open("./data/{}/handled/{}_B_pca128.pkl".format(args.dataset, args.llm_emb_file), "rb"))
        llm_emb_all = pickle.load(open("./data/{}/handled/{}_all.pkl".format(args.dataset, args.llm_emb_file), "rb"))
        #llm_emb_all = np.concatenate([llm_emb_A, llm_emb_B])
        
        llm_item_emb = np.concatenate([
            np.zeros((1, llm_emb_all.shape[1])),
            llm_emb_all
        ])
        if args.global_emb:
            self.item_emb_llm = nn.Embedding.from_pretrained(torch.Tensor(llm_item_emb), padding_idx=0)
        else:
            self.item_emb_llm = nn.Embedding(self.item_numA+self.item_numB+1, args.hidden_size, padding_idx=0)
        if args.freeze_emb:
            self.item_emb_llm.weight.requires_grad = False
        else:
            self.item_emb_llm.weight.requires_grad = True
        self.adapter = nn.Sequential(
            nn.Linear(llm_item_emb.shape[1], int(llm_item_emb.shape[1] / 2)),
            nn.Linear(int(llm_item_emb.shape[1] / 2), args.hidden_size)
        )

        # for mixed sequence
        # self.item_emb = nn.Embedding(self.item_num+1, args.hidden_size, padding_idx=0)
        self.pos_emb = nn.Embedding(args.max_len+1, args.hidden_size)
        self.emb_dropout = nn.Dropout(p=args.dropout_rate)
        self.backbone = SASRecBackbone(device, args)
        self.mi_extractor = MultiInterestExtractor(args)
        
        # --- 修改: 定义拟合影响力的模块和损失 ---
        # 这两个模块将学习去预测由影响函数（梯度）计算出的权重
        self.sensitivity_calculator_A = nn.Linear(args.hidden_size * 2, 1)
        self.sensitivity_calculator_B = nn.Linear(args.hidden_size * 2, 1)
        
        # 用于拟合影响力的损失函数
        self.influence_loss_func = nn.MSELoss()
        # 平衡主任务损失和影响力拟合损失的超参数
        self.gamma = args.gamma if hasattr(args, 'gamma') else 1.0


        # for domain A
        if args.local_emb:
            llm_embA = np.concatenate([np.zeros((1, llm_emb_A.shape[1])), llm_emb_A])
            self.item_embA = nn.Embedding.from_pretrained(torch.Tensor(llm_embA), padding_idx=0)
        else:
            self.item_embA = nn.Embedding(self.item_numA+1, args.hidden_size, padding_idx=0)
        self.pos_embA = nn.Embedding(args.max_len+1, args.hidden_size)
        self.emb_dropoutA = nn.Dropout(p=args.dropout_rate)
        self.backboneA = SASRecBackbone(device, args)

        # for domain B
        if args.local_emb:
            llm_embB = np.concatenate([np.zeros((1, llm_emb_B.shape[1])), llm_emb_B])
            self.item_embB = nn.Embedding.from_pretrained(torch.Tensor(llm_embB), padding_idx=0)
        else:
            self.item_embB = nn.Embedding(self.item_numB+1, args.hidden_size, padding_idx=0)
        self.pos_embB = nn.Embedding(args.max_len+1, args.hidden_size)
        self.emb_dropoutB = nn.Dropout(p=args.dropout_rate)
        self.backboneB = SASRecBackbone(device, args)

        self.loss_func = nn.BCEWithLogitsLoss(reduction="none")

        if args.global_emb: # if use the LLM embedding, do not initilize
            self.filter_init_modules.append("item_emb_llm")
        if args.local_emb:
            self.filter_init_modules.append("item_embA")
            self.filter_init_modules.append("item_embB")
        self._init_weights()


    def _get_embedding(self, log_seqs, domain="A"):

        if domain == "A":
            item_seq_emb = self.item_embA(log_seqs)
        elif domain == "B":
            item_seq_emb = self.item_embB(log_seqs)
        elif domain == "AB":
            if self.global_emb:
                item_seq_emb = self.item_emb_llm(log_seqs)
                item_seq_emb = self.adapter(item_seq_emb)
            else:
                item_seq_emb = self.item_emb_llm(log_seqs)
        else:
            raise ValueError

        return item_seq_emb




    def log2feats(self, log_seqs, positions, domain="A"):

        if domain == "AB":
            seqs = self._get_embedding(log_seqs, domain=domain)
            seqs *= self.item_emb_llm.embedding_dim ** 0.5
            seqs += self.pos_emb(positions.long())
            seqs = self.emb_dropout(seqs)

            log_feats = self.backbone(seqs, log_seqs)

        elif domain == "A":
            seqs = self._get_embedding(log_seqs, domain=domain)
            seqs *= self.item_embA.embedding_dim ** 0.5
            seqs += self.pos_embA(positions.long())
            seqs = self.emb_dropoutA(seqs)

            log_feats = self.backboneA(seqs, log_seqs)

        elif domain == "B":
            seqs = self._get_embedding(log_seqs, domain=domain)
            seqs *= self.item_embB.embedding_dim ** 0.5
            seqs += self.pos_embB(positions.long())
            seqs = self.emb_dropoutB(seqs)

            log_feats = self.backboneB(seqs, log_seqs)

        return log_feats



    def forward(self, 
                seq, pos, neg, positions,
                seqA, posA, negA, positionsA,
                seqB, posB, negB, positionsB,
                target_domain, domain_mask,
                **kwargs):
        '''apply the seq-to-seq loss'''

        # for mixed sequence
        log_feat = self.log2feats(seq, positions, domain="AB")

        topk_weights, topk_indices = self.mi_extractor(log_feat)
        c_u = self.mi_extractor.C(topk_indices) # (B, L, K, H)

        # 原始兴趣贡献
        base_interest_contributions = topk_weights.unsqueeze(-1) * c_u # (B, L, K, H)
        # --- 阶段1: 计算影响力目标 (理想权重) ---
        
        # 1a. 计算初步的域损失
        log_feats_base = base_interest_contributions.sum(dim=2) # (B, L, H)
        
        pos_embs = self._get_embedding(pos, domain="AB")
        neg_embs = self._get_embedding(neg, domain="AB")
        
        pos_logits_base = (log_feats_base * pos_embs).sum(dim=-1)
        neg_logits_base = (log_feats_base * neg_embs).sum(dim=-1)

        log_featsA_base = self.log2feats(seqA, positionsA, domain="A")
        pos_embsA = self._get_embedding(posA, domain="A")
        neg_embsA = self._get_embedding(negA, domain="A")
        pos_logitsA_base = (log_featsA_base * pos_embsA).sum(dim=-1)
        neg_logitsA_base = (log_featsA_base * neg_embsA).sum(dim=-1)
        pos_logitsA_base[posA>0] += pos_logits_base[domain_mask==0]
        neg_logitsA_base[posA>0] += neg_logits_base[domain_mask==0]

        log_featsB_base = self.log2feats(seqB, positionsB, domain="B")
        pos_embsB = self._get_embedding(posB, domain="B")
        neg_embsB = self._get_embedding(negB, domain="B")
        pos_logitsB_base = (log_featsB_base * pos_embsB).sum(dim=-1)
        neg_logitsB_base = (log_featsB_base * neg_embsB).sum(dim=-1)
        pos_logitsB_base[posB>0] += pos_logits_base[domain_mask==1]
        neg_logitsB_base[posB>0] += neg_logits_base[domain_mask==1]

        indicesA = (posA != 0)
        indicesB = (posB != 0)
        lossA_base = self.loss_func(pos_logitsA_base[indicesA], torch.ones_like(pos_logitsA_base[indicesA])) + \
                     self.loss_func(neg_logitsA_base[indicesA], torch.zeros_like(neg_logitsA_base[indicesA]))
        lossB_base = self.loss_func(pos_logitsB_base[indicesB], torch.ones_like(pos_logitsB_base[indicesB])) + \
                     self.loss_func(neg_logitsB_base[indicesB], torch.zeros_like(neg_logitsB_base[indicesB]))

        # 1b. 计算影响力梯度
        # --- 修正: 直接对 base_interest_contributions 求导 ---
        # detach()确保这些梯度不会影响到模型主干的更新
        base_interest_contributions.requires_grad_(True)
        
        grad_A = torch.autograd.grad(lossA_base.sum(), base_interest_contributions, retain_graph=True)[0]
        grad_B = torch.autograd.grad(lossB_base.sum(), base_interest_contributions, retain_graph=True)[0]

        # 1c. 计算理想权重 (target_weights)
        # 梯度为负表示增加贡献可以降低损失，是我们想要的
        # 取梯度的负值并归一化作为理想权重
        target_weights_A = torch.softmax(-torch.norm(grad_A, p=2, dim=-1), dim=-1) # (B, L, K)
        target_weights_B = torch.softmax(-torch.norm(grad_B, p=2, dim=-1), dim=-1) # (B, L, K)

        # --- 阶段2: 拟合影响力权重 ---
        
        # 2a. 预测权重
        user_rep = log_feat.unsqueeze(2).expand_as(c_u)
        interest_rep = c_u
        predicted_scores_A = self.sensitivity_calculator_A(torch.cat([user_rep, interest_rep], dim=-1)).squeeze(-1)
        predicted_scores_B = self.sensitivity_calculator_B(torch.cat([user_rep, interest_rep], dim=-1)).squeeze(-1)
        
        predicted_weights_A = torch.softmax(predicted_scores_A, dim=-1) # (B, L, K)
        predicted_weights_B = torch.softmax(predicted_scores_B, dim=-1) # (B, L, K)

        # 2b. 计算影响力拟合损失
        # 我们希望预测的权重能接近理想权重
        influence_loss = self.influence_loss_func(predicted_weights_A, target_weights_A.detach()) + \
                         self.influence_loss_func(predicted_weights_B, target_weights_B.detach())

        # --- 阶段3: 计算主任务损失 ---
        
        # 3a. 使用预测的权重计算最终的logits
        log_feats_for_A = (predicted_weights_A.unsqueeze(-1) * base_interest_contributions).sum(dim=2)
        log_feats_for_B = (predicted_weights_B.unsqueeze(-1) * base_interest_contributions).sum(dim=2)
        
        pos_logits_for_A = (log_feats_for_A * pos_embs).sum(dim=-1)
        neg_logits_for_A = (log_feats_for_A * neg_embs).sum(dim=-1)
        pos_logits_for_B = (log_feats_for_B * pos_embs).sum(dim=-1)
        neg_logits_for_B = (log_feats_for_B * neg_embs).sum(dim=-1)

        # 3b. 计算最终的域损失
        pos_logitsA = (log_featsA_base * pos_embsA).sum(dim=-1)
        neg_logitsA = (log_featsA_base * neg_embsA).sum(dim=-1)
        pos_logitsA[posA>0] += pos_logits_for_A[domain_mask==0]
        neg_logitsA[posA>0] += neg_logits_for_A[domain_mask==0]

        pos_logitsB = (log_featsB_base * pos_embsB).sum(dim=-1)
        neg_logitsB = (log_featsB_base * neg_embsB).sum(dim=-1)
        pos_logitsB[posB>0] += pos_logits_for_B[domain_mask==1]
        neg_logitsB[posB>0] += neg_logits_for_B[domain_mask==1]

        lossA = self.loss_func(pos_logitsA[indicesA], torch.ones_like(pos_logitsA[indicesA])) + \
                self.loss_func(neg_logitsA[indicesA], torch.zeros_like(neg_logitsA[indicesA]))
        lossB = self.loss_func(pos_logitsB[indicesB], torch.ones_like(pos_logitsB[indicesB])) + \
                self.loss_func(neg_logitsB[indicesB], torch.zeros_like(neg_logitsB[indicesB]))

        # 混合域损失
        indices = (pos != 0)
        lossAB = self.loss_func(pos_logits_base[indices], torch.ones_like(pos_logits_base[indices])) + \
                 self.loss_func(neg_logits_base[indices], torch.zeros_like(neg_logits_base[indices]))

        # 3c. 组合总损失
        main_loss = lossA.mean() + lossB.mean() + lossAB.mean()
        total_loss = main_loss + self.gamma * influence_loss

        return total_loss
    


    def predict(self,
                seq, item_indices, positions,
                seqA, item_indicesA, positionsA,
                seqB, item_indicesB, positionsB,
                target_domain,
                **kwargs): # for inference
        '''Used to predict the score of item_indices given log_seqs'''

        log_feat = self.log2feats(seq, positions, domain="AB")
        topk_weights, topk_indices = self.mi_extractor(log_feat)
        c_u = self.mi_extractor.C(topk_indices)

        # --- 在推理时，直接使用训练好的calculator来预测权重 ---
        last_log_feat = log_feat[:, -1, :]
        last_c_u = c_u[:, -1, :, :]
        last_topk_weights = topk_weights[:, -1, :]

        user_rep = last_log_feat.unsqueeze(1).expand_as(last_c_u)
        interest_rep = last_c_u
        
        predicted_scores_A = self.sensitivity_calculator_A(torch.cat([user_rep, interest_rep], dim=-1)).squeeze(-1)
        predicted_scores_B = self.sensitivity_calculator_B(torch.cat([user_rep, interest_rep], dim=-1)).squeeze(-1)
        
        predicted_weights_A = torch.softmax(predicted_scores_A, dim=-1)
        predicted_weights_B = torch.softmax(predicted_scores_B, dim=-1)

        base_interest_contributions = last_topk_weights.unsqueeze(-1) * last_c_u
        final_feat_for_A = (predicted_weights_A.unsqueeze(-1) * base_interest_contributions).sum(dim=1)
        final_feat_for_B = (predicted_weights_B.unsqueeze(-1) * base_interest_contributions).sum(dim=1)
        
        item_embs = self._get_embedding(item_indices, domain="AB")
        
        logits_for_A = item_embs.matmul(final_feat_for_A.unsqueeze(-1)).squeeze(-1)
        logits_for_B = item_embs.matmul(final_feat_for_B.unsqueeze(-1)).squeeze(-1)
        
        # for domain A
        log_featsA = self.log2feats(seqA, positionsA, domain="A")
        final_featA = log_featsA[:, -1, :]
        item_embsA = self._get_embedding(item_indicesA, domain="A")
        logitsA = item_embsA.matmul(final_featA.unsqueeze(-1)).squeeze(-1)

        # for domain B
        log_featsB = self.log2feats(seqB, positionsB, domain="B")
        final_featB = log_featsB[:, -1, :]
        item_embsB = self._get_embedding(item_indicesB, domain="B")
        logitsB = item_embsB.matmul(final_featB.unsqueeze(-1)).squeeze(-1)

        logits_A_combined = logitsA + logits_for_A
        logits_B_combined = logitsB + logits_for_B
        
        final_logits = torch.where(target_domain.unsqueeze(-1) == 0, logits_A_combined, logits_B_combined)

        return final_logits




class CMGR(CMGR_base):

    def __init__(self, user_num, item_num_dict, device, args):
        super().__init__(user_num, item_num_dict, device, args)
        # 只保留基础初始化，完全去除reg_loss相关内容
        self._init_weights()


    def forward(self, 
                seq, pos, neg, positions,
                seqA, posA, negA, positionsA,
                seqB, posB, negB, positionsB,
                target_domain, domain_mask,
                reg_A=None, reg_B=None,
                user_id=None,
                **kwargs):
        # 只保留主损失
        loss = super().forward(seq, pos, neg, positions,
                    seqA, posA, negA, positionsA,
                    seqB, posB, negB, positionsB,
                    target_domain, domain_mask,
                    **kwargs)
        return loss


