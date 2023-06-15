import timm
import torch


def patch_vit_fix_block_num(vit_model: timm.models.vision_transformer.VisionTransformer, num_blocks: int):
    '''
    Patch forward path of Vision Transformer.
    Other than the number of blocks, the model is the same as the original ViT.
    '''

    def forward_features(self, x, threshold: float = 0.5):
        x = self.patch_embed(x)
        x = self._pos_embed(x)
        x = self.patch_drop(x)
        x = self.norm_pre(x)
        if self.grad_checkpointing and not torch.jit.is_scripting():
            x = timm.models.checkpoint_seq(self.blocks, x)
        else:
            x = self.blocks[:num_blocks](x)
        x = self.norm(x)
        return x
    
    vit_model.forward_features = forward_features.__get__(vit_model, timm.models.vision_transformer.VisionTransformer)
    return vit_model


def patch_vit_fix_block_num_confidence(vit_model: timm.models.vision_transformer.VisionTransformer, num_blocks: int, threshold: int):
    '''
    Patch forward path of Vision Transformer.
    Other than the number of blocks, the model is the same as the original ViT.
    '''

    def forward_features(self, x):
        x = self.patch_embed(x)
        x = self._pos_embed(x)
        x = self.patch_drop(x)
        x = self.norm_pre(x)
        if self.grad_checkpointing and not torch.jit.is_scripting():
            x = timm.models.checkpoint_seq(self.blocks, x)
        else:
            x = self.blocks[:num_blocks](x)
            output = self.forward_head(self.norm(x))
            confidence  = torch.softmax(output, dim=-1).max(dim=-1)[0].min()
            if confidence < threshold:
                x = self.blocks[num_blocks:](x)
            else:
                return output, "pre-exit"
        x = self.norm(x)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        if type(x) == tuple: # pre-exit
            return x[0]
        x = self.forward_head(x)
        return x
    
    vit_model.forward_features = forward_features.__get__(vit_model, timm.models.vision_transformer.VisionTransformer)
    vit_model.forward = forward.__get__(vit_model, timm.models.vision_transformer.VisionTransformer)
    return vit_model
