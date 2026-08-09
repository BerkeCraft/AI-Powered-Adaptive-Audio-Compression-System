import torch
from encodec import EncodecModel

print("Testing EnCodec...")
model = EncodecModel.encodec_model_24khz()
model.set_target_bandwidth(6.0)
print("Model loaded")

# Create a dummy audio tensor: 1 second of silence at 24kHz, mono
dummy_audio = torch.zeros(1, 1, 24000)  # (channels, samples)
print("Dummy audio shape:", dummy_audio.shape)

# EnCodec expects (batch, channels, samples)
dummy_audio = dummy_audio.unsqueeze(0)  # (1, 1, 24000)
print("Input shape to encode:", dummy_audio.shape)

encoded_frames = model.encode(dummy_audio)
print("Encoded frames type:", type(encoded_frames))
print("Encoded frames length:", len(encoded_frames))
if len(encoded_frames) > 0:
    print("First frame type:", type(encoded_frames[0]))
    if isinstance(encoded_frames[0], tuple):
        print("First frame[0] type:", type(encoded_frames[0][0]))
        print("First frame[0] shape:", encoded_frames[0][0].shape)
        print("First frame[1] type:", type(encoded_frames[0][1]))
        print("First frame[1] shape:", encoded_frames[0][1].shape)
else:
    print("Encoded frames is empty!")