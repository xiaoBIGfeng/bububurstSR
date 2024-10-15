            burst_feat_flatten = burst_feat.view(n, -1)
            x_fc = self.fc(burst_feat_flatten)
            prob = torch.sigmoid(x_fc)
            _, indices = torch.topk(prob.view(-1), k=8, dim=0)
            select_frame = torch.index_select(burst_feat, 0, indices)
            burst_feat = select_frame.view(8, 48, h, w)
  File "/mnt/diskb/penglong/anaconda3/envs/dx9826/lib/python3.9/site-packages/torch/nn/modules/linear.py", line 114, in forward
    return F.linear(input, self.weight, self.bias)
RuntimeError: mat1 and mat2 shapes cannot be multiplied (14x110592 and 7077888x1)
