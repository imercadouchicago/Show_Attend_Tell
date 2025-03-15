import torch
from torch import nn
from torchvision.models import resnet101, vgg19

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Encoder(nn.Module):
    def __init__(self, network='vgg19'):
        super(Encoder, self).__init__()
        self.network = network
        if network == 'resnet101':
            # Loads a pre-trained ResNet-101 model
            self.net = resnet101(pretrained=True)  
            # Removes the last two layers (avg pooling and FC) to get feature maps
            self.net = nn.Sequential(*list(self.net.children())[:-2])
            # Sets the output dimension to 2048 (number of channels in ResNet-101's final conv layer)
            self.dim = 2048
        elif network == 'vgg19':
            # Loads a pre-trained VGG19 model
            self.net = vgg19(pretrained=True)  
            # Removes the last layer to get feature maps
            self.net = nn.Sequential(*list(self.net.features.children())[:-1])  
            # Sets the output dimension to 512 (number of channels in VGG19's final conv layer)
            self.dim = 512

    def forward(self, image):
        # Passes the input through the CNN to get feature maps
        output = self.net(image) 
        # Rearranges dimensions from (batch, channels, height, width) to (batch, height, width, channels)
        output = output.permute(0, 2, 3, 1) 
        # Reshapes tensor to (batch, height*width, channels) - flattening spatial dimensions
        output = output.view(output.size(0), -1, output.size(-1))  
        return output
    
    def fine_tune(self, fine_tune=True):
        """
        Allow or prevent the computation of gradients for convolutional blocks 2 through 4 of the encoder.
        """
        for p in self.resnet.parameters():
            p.requires_grad = False
        # If fine-tuning, only fine-tune convolutional blocks 2 through 4
        for c in list(self.resnet.children())[5:]:
            for p in c.parameters():
                p.requires_grad = fine_tune

    
class Attention(nn.Module):
    """
    Soft attention - uses softmax to create a probability distribution over spatial locations
    """
    def __init__(self, encoder_dim, decoder_dim=512, attention_dim=512):
        super(Attention, self).__init__()
        self.decoder_proj = nn.Linear(decoder_dim, attention_dim)
        self.encoder_proj = nn.Linear(encoder_dim, attention_dim)
        self.final_proj = nn.Linear(attention_dim, 1)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, img_features, hidden_state):
        # Transforms hidden state and adds a dimension to match img_features shape
        U_h = self.decoder_proj(hidden_state).unsqueeze(1)  
        # Transforms image features
        W_a = self.encoder_proj(img_features)  
        # Combines transformed features and applies tanh
        att = self.tanh(W_a + U_h)  
        # Computes attention scores and removes last dimension
        e = self.final_proj(att).squeeze(2)  
        # Applies softmax to get attention weights
        alpha = self.softmax(e)  
        # Computes weighted sum of features using attention weights
        context = (img_features * alpha.unsqueeze(2)).sum(dim=1)  
        return context, alpha
    
class Decoder(nn.Module):
    def __init__(self, vocab_size, encoder_dim, tf=False):
        super(Decoder, self).__init__()
        self.vocab_size = vocab_size
        self.encoder_dim = encoder_dim
        self.decoder_dim = 512
        self.attention_dim = 512
        self.embedding_dim = 512

        self.init_h = nn.Linear(encoder_dim, self.decoder_dim)
        self.init_c = nn.Linear(encoder_dim, self.decoder_dim)
        self.attention_gate_layer = nn.Linear(self.decoder_dim, encoder_dim)
        self.output_layer = nn.Linear(self.decoder_dim, self.vocab_size)

        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout()
        
        self.attention = Attention(self.encoder_dim, self.decoder_dim, self.attention_dim)
        self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim)
        self.lstm = nn.LSTMCell(self.embedding_dim + self.encoder_dim, self.decoder_dim)
        self.use_tf = tf  # Whether to use teacher forcing
    
    def init_lstm_states(self, img_features):
        # Computes mean of features across spatial locations
        avg_features = img_features.mean(dim=1)
        
        # Transforms average features to initialize cell state
        c = self.init_c(avg_features)
        c = self.tanh(c)
        
        # Transforms average features to initialize hidden state
        h = self.init_h(avg_features)
        h = self.tanh(h)
        
        return h, c
    
    def forward(self, img_features, captions):
        # img_features is a batch_size x L x D matrix with L annotation vectors of dimension D
        batch_size = img_features.size(0)
        num_pixels = img_features.size(1)
        
        # Sort input data by decreasing caption lengths
        caption_lengths = torch.tensor([len(cap) for cap in captions])
        caption_lengths, sort_ind = caption_lengths.sort(dim=0, descending=True)
        img_features = img_features[sort_ind]
        captions = captions[sort_ind]
        
        h, c = self.init_lstm_states(img_features)
        
        # Initialize starting tokens
        prev_words = torch.zeros(batch_size, 1).long().cuda()
        embedding = self.embedding(prev_words)
        
        # Get maximum caption length minus 1 (to account for end token)
        max_decode_len = max([len(caption) for caption in captions]) - 1
        
        # Create tensors to hold prediction and attention distributions
        predictions = torch.zeros(batch_size, max_decode_len, self.vocab_size).cuda()
        alphas = torch.zeros(batch_size, max_decode_len, num_pixels).cuda()
        
        for t in range(max_decode_len):
            # Calculate how many captions are active at this timestep
            batch_size_t = sum([l > t for l in caption_lengths])
            
            # Adaptive attention where h is previous hidden state
            context, alpha = self.attention(img_features[:batch_size_t], h[:batch_size_t])
            gate = self.sigmoid(self.f_beta(h[:batch_size_t]))
            gated_context = gate * context
            
            if self.use_tf and self.training:
                # Teacher forcing: use ground truth as input for next timestep
                embedding_t = self.embedding(captions[:batch_size_t, t])
                lstm_input = torch.cat((embedding_t, gated_context), dim=1)
            else:
                # Use model's own prediction as input for next timestep
                embedding = embedding.squeeze(1) if embedding.dim() == 3 else embedding
                embedding_t = embedding[:batch_size_t]
                lstm_input = torch.cat((embedding_t, gated_context), dim=1)
            
            h_t, c_t = self.lstm(lstm_input, (h[:batch_size_t], c[:batch_size_t]))
            output = self.deep_output(self.dropout(h_t))
            
            # Update states for next timestep
            h[:batch_size_t] = h_t
            c[:batch_size_t] = c_t
            
            predictions[:batch_size_t, t] = output
            alphas[:batch_size_t, t] = alpha
            
            if not self.training or not self.use_tf:
                embedding = self.embedding(output.max(1)[1].reshape(batch_size_t, 1))
        
        # Return sorted indices to restore original order
        _, restore_ind = sort_ind.sort(dim=0, descending=False)
        return predictions[restore_ind], alphas[restore_ind]

    def caption(self, img_features, beam_size):  # Method for generating captions using beam search
        """
        We use beam search to construct the best sentences following a
        similar implementation as the author in
        https://github.com/kelvinxu/arctic-captions/blob/master/generate_caps.py
        """
        prev_words = torch.zeros(beam_size, 1).long()  # Creates tensor of zeros (start tokens) for beam search

        sentences = prev_words  # Initializes sentences with start tokens
        top_preds = torch.zeros(beam_size, 1)  # Initializes prediction scores
        alphas = torch.ones(beam_size, 1, img_features.size(1))  # Initializes attention weights

        completed_sentences = []  # List to store completed sentences
        completed_sentences_alphas = []  # List to store attention weights for completed sentences
        completed_sentences_preds = []  # List to store prediction scores for completed sentences

        step = 1  # Initializes step counter
        h, c = self.init_lstm_states(img_features)

        while True:  # Continues until breaking condition
            embedding = self.embedding(prev_words).squeeze(1)  # Embeds previous words and removes extra dimension
            context, alpha = self.attention(img_features, h)  # Computes attention context and weights
            gate = self.sigmoid(self.attention_gate_layer(h))  # Computes attention gate
            gated_context = gate * context  # Applies gate to context

            lstm_input = torch.cat((embedding, gated_context), dim=1)  # Concatenates embedding and context
            h, c = self.lstm(lstm_input, (h, c))  # Updates LSTM states
            output = self.output_layer(h)  # Maps LSTM output to vocabulary
            output = top_preds.expand_as(output) + output  # Adds previous scores to current output

            if step == 1:  # Special case for first step
                top_preds, top_words = output[0].topk(beam_size, 0, True, True)  # Gets top k predictions
            else:
                top_preds, top_words = output.view(-1).topk(beam_size, 0, True, True)  # Gets top k predictions from all beams
            prev_word_idxs = top_words / output.size(1)  # Gets indices of which beams to keep
            next_word_idxs = top_words % output.size(1)  # Gets indices of next words

            sentences = torch.cat((sentences[prev_word_idxs], next_word_idxs.unsqueeze(1)), dim=1)  # Updates sentences
            alphas = torch.cat((alphas[prev_word_idxs], alpha[prev_word_idxs].unsqueeze(1)), dim=1)  # Updates attention weights

            incomplete = [idx for idx, next_word in enumerate(next_word_idxs) if next_word != 1]  # Finds incomplete sentences
            complete = list(set(range(len(next_word_idxs))) - set(incomplete))  # Finds completed sentences

            if len(complete) > 0:  # If there are completed sentences
                completed_sentences.extend(sentences[complete].tolist())  # Adds completed sentences to list
                completed_sentences_alphas.extend(alphas[complete].tolist())  # Adds attention weights to list
                completed_sentences_preds.extend(top_preds[complete])  # Adds prediction scores to list
            beam_size -= len(complete)  # Reduces beam size

            if beam_size == 0:  # If no more beams
                break  # Exits the loop
            sentences = sentences[incomplete]  # Keeps only incomplete sentences
            alphas = alphas[incomplete]  # Keeps only corresponding attention weights
            h = h[prev_word_idxs[incomplete]]  # Updates hidden states
            c = c[prev_word_idxs[incomplete]]  # Updates cell states
            img_features = img_features[prev_word_idxs[incomplete]]  # Updates image features
            top_preds = top_preds[incomplete].unsqueeze(1)  # Updates prediction scores
            prev_words = next_word_idxs[incomplete].unsqueeze(1)  # Updates previous words

            if step > 50:  # If exceeded maximum steps
                break  # Exits the loop
            step += 1  # Increments step counter

        idx = completed_sentences_preds.index(max(completed_sentences_preds))  # Finds index of best sentence
        sentence = completed_sentences[idx]  # Gets the best sentence
        alpha = completed_sentences_alphas[idx]  # Gets corresponding attention weights
        return sentence, alpha  # Returns the best sentence and its attention weights
