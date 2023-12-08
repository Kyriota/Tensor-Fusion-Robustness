import torch



class ModelType:
    TF = 'tensor_fusion'
    CONCAT = 'concat'
    ETF = 'efficient_tensor_fusion'
    MULTI_CONCAT = 'multi_concat'
    MULTIPLY = 'multiplication'



# A simple network for MNIST
class SimpleNet(torch.nn.Module):
    def __init__(self, model_type=ModelType.CONCAT, emb_len=8):
        super(SimpleNet, self).__init__()

        self.emb_len = emb_len
        self.model_type = model_type

        # 14 * 14 blocks
        self.fc1_lt = self.create_block() # left top
        self.fc1_rt = self.create_block() # right top
        self.fc1_lb = self.create_block() # left bottom
        self.fc1_rb = self.create_block() # right bottom

        if self.model_type == ModelType.TF:
            # tensor fusion network
            self.fc2 = torch.nn.Linear((self.emb_len+1) ** 4, 32)
        elif self.model_type == ModelType.CONCAT:
            # concat network
            self.fc2 = torch.nn.Linear(self.emb_len * 4, 32)
        elif self.model_type == ModelType.MULTI_CONCAT:
            # multi concat network
            self.repeat_times = int((self.emb_len+1)**4 / self.emb_len / 4)
            self.fc2 = torch.nn.Linear(self.emb_len * self.repeat_times * 4, 32)
        elif self.model_type == ModelType.MULTIPLY:
            # multiplication network
            self.fc2 = torch.nn.Linear(self.emb_len, 32)
        elif self.model_type == ModelType.ETF:
            # efficient tensor fusion network
            rank, out_dim = 32, 64
            self.w_lt = torch.nn.parameter.Parameter(torch.Tensor(rank, self.emb_len+1, out_dim))
            self.w_rt = torch.nn.parameter.Parameter(torch.Tensor(rank, self.emb_len+1, out_dim))
            self.w_lb = torch.nn.parameter.Parameter(torch.Tensor(rank, self.emb_len+1, out_dim))
            self.w_rb = torch.nn.parameter.Parameter(torch.Tensor(rank, self.emb_len+1, out_dim))
            self.fusion_w = torch.nn.parameter.Parameter(torch.Tensor(1, rank))
            self.fusion_b = torch.nn.parameter.Parameter(torch.Tensor(1, out_dim))

            torch.nn.init.kaiming_normal_(self.w_lt.unsqueeze(0))
            torch.nn.init.kaiming_normal_(self.w_rt.unsqueeze(0))
            torch.nn.init.kaiming_normal_(self.w_lb.unsqueeze(0))
            torch.nn.init.kaiming_normal_(self.w_rb.unsqueeze(0))
            torch.nn.init.kaiming_normal_(self.fusion_w.unsqueeze(0))
            self.fusion_b.data.fill_(0)

            self.fc2 = torch.nn.Linear(out_dim, 32)
        else:
            raise ValueError('Invalid model type!')

        self.fc3 = torch.nn.Linear(32, 10)

        self.activation = torch.nn.ReLU()
    
    def create_block(self):
        return torch.nn.Sequential(
            torch.nn.Linear(14 * 14, self.emb_len * 2),
            torch.nn.ReLU(),
            torch.nn.Linear(self.emb_len * 2, self.emb_len),
        )

    def forward(self, x):
        # x: (batch_size, 1, 28, 28)
        # get left top, right top, left bottom, right bottom
        lt = x[:, :, :14, :14].reshape(-1, 14 * 14)
        rt = x[:, :, :14, 14:].reshape(-1, 14 * 14)
        lb = x[:, :, 14:, :14].reshape(-1, 14 * 14)
        rb = x[:, :, 14:, 14:].reshape(-1, 14 * 14)

        # 7 * 7 blocks
        lt = self.fc1_lt(lt)
        rt = self.fc1_rt(rt)
        lb = self.fc1_lb(lb)
        rb = self.fc1_rb(rb)

        if self.model_type == ModelType.TF or self.model_type == ModelType.ETF:
            lt = torch.cat([lt, torch.ones(lt.shape[0], 1).to(lt.device)], dim=1)
            rt = torch.cat([rt, torch.ones(rt.shape[0], 1).to(rt.device)], dim=1)
            lb = torch.cat([lb, torch.ones(lb.shape[0], 1).to(lb.device)], dim=1)
            rb = torch.cat([rb, torch.ones(rb.shape[0], 1).to(rb.device)], dim=1)

        if self.model_type == ModelType.TF:
            # outer product of 4 blocks
            x = torch.einsum('bi,bj,bk,bl->bijkl', lt, rt, lb, rb)
            x = x.reshape(-1, (self.emb_len+1) ** 4)
        elif self.model_type == ModelType.CONCAT:
            x = torch.cat([lt, rt, lb, rb], dim=1)
        elif self.model_type == ModelType.MULTI_CONCAT:
            x = torch.cat([lt] * self.repeat_times + [rt] * self.repeat_times + [lb] * self.repeat_times + [rb] * self.repeat_times, dim=1)
        elif self.model_type == ModelType.MULTIPLY:
            x = lt * rt * lb * rb
        else: # self.model_type == ModelType.ETF
            fusion_lt = torch.matmul(lt, self.w_lt).permute(1, 0, 2)
            fusion_rt = torch.matmul(rt, self.w_rt).permute(1, 0, 2)
            fusion_lb = torch.matmul(lb, self.w_lb).permute(1, 0, 2)
            fusion_rb = torch.matmul(rb, self.w_rb).permute(1, 0, 2)

            fusion = fusion_lt * fusion_rt * fusion_lb * fusion_rb
            x = torch.matmul(self.fusion_w, fusion).squeeze() + self.fusion_b
        
        x = self.activation(x)
        x = self.fc2(x)

        x = self.activation(x)
        x = self.fc3(x)

        return x