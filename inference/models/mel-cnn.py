import os
model.load_state_dict(torch.load('best_model.pth'))
model.to(device)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def CNN_predict(audio_path, model,device):
    mel_spec = preprocess_audio(audio_path)
    mel_spec = mel_spec.unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        output = model(mel_spec)
        probs = F.softmax(output, dim=1)
        pred_class = torch.argmax(probs, dim=1).item()
        confidence = probs[0, pred_class].item()
    
    label_map = {0: "REAL", 1: "FAKE"}
    return label_map[pred_class], confidence

file_path = Path("test.wav")
print("Tồn tại không:", os.path.exists(file_path))
print(file_path)
label, prob = CNN_predict(file_path, model, device)
print(f"CNN predicts: {label} ({prob*100:.2f}%)")



