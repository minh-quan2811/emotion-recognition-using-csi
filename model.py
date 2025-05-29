import torch
import torch.nn as nn
from transformers import ViTForImageClassification, AutoModelForImageClassification, ViTConfig

class VisionTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.transformer = ViTForImageClassification(config)

    def forward(self, x):
        output = self.transformer(x, output_hidden_states=True)
        return output.logits, output.hidden_states

    def freeze_layers(self, layers_to_freeze):
        if 0 in layers_to_freeze:
            for param in self.transformer.vit.embeddings.parameters():
                param.requires_grad = False

        # Freeze layers
        for i, layer in enumerate(self.transformer.vit.encoder.layer):
            if i in layers_to_freeze:
                for param in layer.parameters():
                    param.requires_grad = False

    def load_teacher_weights(self, teacher_model):
        teacher_vit = teacher_model.vit
        student_vit = self.transformer.vit
        state = {}
        for k, v in teacher_vit.named_parameters():
            if 'embedding' in k:
                continue
            state[k] = v.detach().clone()

        student_vit.load_state_dict(state, strict=False)

def initialize_models():
    # Teacher model
    model_name = "dima806/facial_emotions_image_detection"
    teacher_model = AutoModelForImageClassification.from_pretrained(model_name).cuda()
    for param in teacher_model.parameters():
        param.requires_grad = False

    # Student model
    config = ViTConfig(
        num_channels=52,
        num_labels=4,
        output_hidden_states=True
    )
    student_model = VisionTransformer(config).cuda()

    # Load teacher weights into student model (excluding embeddings)
    student_model.load_teacher_weights(teacher_model)

    # Optionally freeze layers (commented out in notebook)
    # student_model.freeze_layers(layers_to_freeze=list(range(11)))

    return teacher_model, student_model
