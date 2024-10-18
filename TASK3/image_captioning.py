import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import nltk
from nltk.tokenize import word_tokenize

# Download the NLTK tokenizer
nltk.download('punkt')

# Load the VGG16 model (pretrained on ImageNet)
vgg = models.vgg16(pretrained=True).features
vgg.eval()

# Define preprocessing transformations for the image
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Extract features from an image using VGG16
def extract_image_features(image_path):
    img = Image.open(image_path).convert('RGB')
    img = preprocess(img).unsqueeze(0)
    with torch.no_grad():
        features = vgg(img)
    return features.squeeze()

# Define the RNN model for generating captions
class CaptionRNN(nn.Module):
    def __init__(self, feature_size, hidden_size, vocab_size, num_layers=1):
        super(CaptionRNN, self).__init__()
        self.lstm = nn.LSTM(feature_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)
    
    def forward(self, features, captions):
        # Run the features and captions through the LSTM
        lstm_out, _ = self.lstm(features.unsqueeze(1))
        outputs = self.fc(lstm_out.squeeze(1))
        return outputs

# Sample vocabulary (you would need a larger, realistic one for training)
vocab = {'<start>': 1, '<end>': 2, 'a': 3, 'man': 4, 'riding': 5, 'horse': 6, 'on': 7, 'the': 8, 'beach': 9}

# Function to convert a caption to tokens
def caption_to_tokens(caption):
    tokens = word_tokenize(caption.lower())
    return [vocab.get(word, 0) for word in tokens]

# Set hyperparameters
feature_size = 512 * 7 * 7  # This is the output size of VGG's feature map
hidden_size = 512
vocab_size = len(vocab) + 1

# Initialize the RNN model
model = CaptionRNN(feature_size, hidden_size, vocab_size)

# Test the pipeline
if __name__ == "__main__":
    # Step 1: Extract features from an image
    image_features = extract_image_features('sample_image.jpg')
    print(f"Extracted image features: {image_features.shape}")
    
    # Step 2: Tokenize a sample caption
    caption = "A man riding a horse on the beach."
    tokens = caption_to_tokens(caption)
    print(f"Tokenized caption: {tokens}")
    
    # Step 3: Run the features through the RNN to generate a caption
    tokens_tensor = torch.tensor(tokens).unsqueeze(0)  # Batch dimension
    output = model(image_features, tokens_tensor)
    print(f"Model output: {output.shape}")
