import copy
import Levenshtein
import torch
import torch.nn.functional as F

class GPROTrainer:
    def __init__(self, model, optimizer, epsilon=0.2, beta=0.1, sync_interval=5000,device='cuda'):
        self.model = model
        self.optimizer = optimizer
        self.reference_model = copy.deepcopy(model)
        self.reference_model.eval()
        for p in self.reference_model.parameters():
            p.requires_grad_(False)
            
        self.epsilon = epsilon
        self.beta = beta
        self.sync_interval = sync_interval
        self.steps = 0
        self.device = device
        
    def compute_reward(self, preds, targets):
        rewards = []
        for pred, target in zip(preds, targets):
            pred_str = ''.join([chr(c) for c in pred if c > 2])
            target_str = ''.join([chr(c) for c in target if c > 2])
            distance = Levenshtein.distance(pred_str, target_str)
            rewards.append(1.0 / (1.0 + distance))
        return torch.tensor(rewards, device=self.device)

    def update(self, batch):
        self.model.train()
        images, targets, _ = batch
        images = images.to(self.device)
        targets = targets.to(self.device)
        
        
        with torch.no_grad():
            _, preds = self.model(images)
        
        
        rewards = self.compute_reward(preds, targets)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
        
        
        online_logits = self.model(images, tgt_tokens=targets)
        with torch.no_grad():
            ref_logits = self.reference_model(images, tgt_tokens=targets)
            
        # Compute loss
        loss = self.gpro_loss(
            online_logits, 
            ref_logits, 
            targets[:,1:],  # Удаляем SOS токен
            rewards
        )
        
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Синхронизация моделей(Думаю, не пригодится)
        if self.steps % self.sync_interval == 0:
            self.reference_model.load_state_dict(self.model.state_dict())
        self.steps += 1
        
        return loss.item()
    
    def gpro_loss(self, logits, old_logits, actions, rewards):
        probs = torch.softmax(logits, dim=-1)  # (B, T-1, V)
        old_probs = torch.softmax(old_logits, dim=-1).detach()  # (B, T-1, V)
        
        # Исправляем размерности
        actions = actions.long()  # Убедимся, что индексы целочисленные
        ratio = probs.gather(-1, actions.unsqueeze(-1)).squeeze(-1) / (  # (B, T-1)
                old_probs.gather(-1, actions.unsqueeze(-1)).squeeze(-1) + 1e-8)
        
        # Преобразуем rewards к правильной размерности
        rewards = rewards.unsqueeze(-1)  # (B, 1)
        clipped = torch.clamp(ratio, 1-self.epsilon, 1+self.epsilon)
        
        # Вычисляем loss с правильной размерностью
        policy_loss = -torch.min(ratio * rewards, clipped * rewards).mean()
        
        
        kl_loss = F.kl_div(
            F.log_softmax(logits, dim=-1),
            F.softmax(old_logits, dim=-1),
            reduction='batchmean'
        )
        
        return policy_loss + self.beta * kl_loss