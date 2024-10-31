import torch
import timm
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
import warnings
import torch.nn.functional as F

warnings.filterwarnings("ignore", category=FutureWarning)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
modelo_path = 'C:/Users/JANDERSON/Desktop/TRABALHO-IA/models-and-bot/best_model.pth'
carregar = torch.load(modelo_path, map_location=device)
model = timm.create_model('efficientnet_b0.ra_in1k', pretrained=False)
model.classifier = nn.Linear(model.classifier.in_features, 2) 
model.load_state_dict(carregar['model'])
model.to(device)
model.eval()

# Função de pré-processamento para imagens
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Me mande uma imagem de controle de xbox  🎮 ou uma abóbora de HalloWeen 🎃")

def classificar(image_path):
    image = Image.open(image_path).convert("RGB")
    x = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(x)
        prob = F.softmax(output, dim=1)
        confiabilidade, predicted = torch.max(prob, 1)

    if confiabilidade.item() < 0.8:
        return -1

    return predicted.item()

async def identificar(update: Update, context: ContextTypes.DEFAULT_TYPE):

    photo = await update.message.photo[-1].get_file()
    image_path = "./images-bot/temp.jpg"
    await photo.download_to_drive(image_path)
    
    label = classificar(image_path)

    if label == 0:
        await update.message.reply_text("É uma abóbora de halloween 🎃")
        print(f"Previu uma abobora")
    elif label == 1:
        await update.message.reply_text("É um controle de xbox 🎮")
        print(f"Previu um controle de xbox")
    else:
        await update.message.reply_text("Não sei que imagem é essa 🤔")

def main():
    app = Application.builder().token("7983786530:AAE905eq2VZ-xcWzrmXXPas5czueWwzUTj4").build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.PHOTO, identificar))

    app.run_polling()

if __name__ == '__main__':
    main()