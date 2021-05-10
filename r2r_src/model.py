import torch
import torch.nn as nn
from param import args

class Lang_Vision_Encoder(nn.Module):
    """ Bidirectional LSTM that encodes the given navigational instruction
    and updates the state representation """

    def __init__(self, input_len, embedding_dim, hidden_dim, padding_idx, dropout_ratio=0.2, bidirectional=False,
                 num_layers=2):
        super(Lang_Vision_Encoder, self).__init__()
        self.input_len = input_len
        self.hidden_size = hidden_dim
        self.num_layers = num_layers
        self.lstm_directions = 2 if bidirectional else 1
        self.word_embeds = nn.Embedding(input_len, embedding_dim, padding_idx)
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim,
                            num_layers=num_layers, batch_first=True, dropout=dropout_ratio,
                            bidirectional=bidirectional)
        self.init_decoder_seq = nn.Sequential(nn.Linear(self.hidden_size * self.lstm_directions,
                                                        self.hidden_size * self.lstm_directions), nn.Tanh())
        self.init_graph_seq = nn.Sequential(nn.Linear(self.hidden_size * self.lstm_directions,
                                                      self.hidden_size * self.lstm_directions), nn.Tanh())

    def forward(self, input_seq, lens):
        """
        :param input_seq: Input vocabulary
        :param lens: Length of dynamic batch sizes
        :return: Hidden state, decoder initial state, cell states and initial graph state
        """
        ''' Initialize the first hidden and cell state to zero'''
        initial_hs = torch.zeros(self.num_layers * self.lstm_directions, input_seq.size(0), self.hidden_size).cuda()
        initial_cs = torch.zeros(self.num_layers * self.lstm_directions, input_seq.size(0), self.hidden_size).cuda()

        ''' Converts the given instruction with a learned embedding'''
        x = self.word_embeds(input_seq)
        x_embeds = nn.utils.rnn.pack_padded_sequence(x, lens, batch_first=True)

        ''' Apply a bidirectional-LSTM to encode the entire sentence'''
        output, (h_n, c_n) = self.lstm(x_embeds, (initial_hs, initial_cs))
        output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)

        c_t = c_n[-1]
        if self.lstm_directions == 2:
            c_t = torch.cat((c_n[-1], c_n[-2]), 1)

        max_out, _ = output.max(1)
        initial_decoder = self.init_decoder_seq(max_out)
        initial_graph = self.init_graph_seq(max_out)
        return output, initial_decoder, c_t, initial_graph


class Language_Attention_Cxt(nn.Module):
    """ Node Specialized Context. Performs three soft attentions independently on instruction
     using the current agent state to give scene, object and action specialized attended contexts"""

    def __init__(self, hidden_size):
        super(Language_Attention_Cxt, self).__init__()
        self.soft_attention_scene = SoftAttention(hidden_size, hidden_size)
        self.soft_attention_object = SoftAttention(hidden_size, hidden_size)
        self.soft_attention_action = SoftAttention(hidden_size, hidden_size)

    def forward(self, h_t, instr, ctx_mask, full_feature_len):
        """ Returns the specialized contexts and the global context which is the
        average of the three specialized contexts"""
        h_tilde_S, _ = self.soft_attention_scene(h_t, instr, ctx_mask, output_tilde=False)
        h_tilde_O, _ = self.soft_attention_object(h_t, instr, ctx_mask, output_tilde=False)
        h_tilde_A, _ = self.soft_attention_action(h_t, instr, ctx_mask, output_tilde=False)

        ''' Attended global context'''
        h_tilde_G = (h_tilde_S + h_tilde_O + h_tilde_A) / 3.0

        cxt_scene = torch.repeat_interleave(h_tilde_S.unsqueeze(1), full_feature_len, dim=1)
        cxt_object = torch.repeat_interleave(h_tilde_O.unsqueeze(1), full_feature_len, dim=1)
        cxt_action = torch.repeat_interleave(h_tilde_A.unsqueeze(1), full_feature_len, dim=1)

        return h_tilde_G, h_tilde_S, h_tilde_O, h_tilde_A, cxt_scene, cxt_object, cxt_action


class Language_Attention_Relational_Cxt(nn.Module):
    """ Edge relational contexts. Constructs relational contexts between action-scene,
    scene-object and object-action specialized contexts. Also applies a learnable non-linear projection,
    with Tanh as the activation function """

    def __init__(self, hidden_size):
        super(Language_Attention_Relational_Cxt, self).__init__()
        self.h_as = nn.Sequential(nn.Linear(hidden_size * 2, hidden_size, bias=False), nn.Tanh())
        self.h_so = nn.Sequential(nn.Linear(hidden_size * 2, hidden_size, bias=False), nn.Tanh())
        self.h_oa = nn.Sequential(nn.Linear(hidden_size * 2, hidden_size, bias=False), nn.Tanh())

        self.soft_attention_as = SoftAttention(hidden_size, hidden_size)
        self.soft_attention_so = SoftAttention(hidden_size, hidden_size)
        self.soft_attention_oa = SoftAttention(hidden_size, hidden_size)

    def forward(self, h_tilde_s, h_tilde_o, h_tilde_a, enc_out, full_feature_len):
        """

        :param h_tilde_s: Specialized scene context
        :param h_tilde_o: Specialized object context
        :param h_tilde_a: Specialized action context
        :param enc_out: Encoder output
        :param full_feature_len: Length of image and pano object features
        :return: Relational contexts
        """
        h_tilde_as = self.h_as(torch.cat((h_tilde_a, h_tilde_s), dim=1))
        h_tilde_so = self.h_so(torch.cat((h_tilde_s, h_tilde_o), dim=1))
        h_tilde_oa = self.h_oa(torch.cat((h_tilde_o, h_tilde_a), dim=1))

        soft_attn_as, _ = self.soft_attention_as(h_tilde_as, enc_out)
        soft_attn_so, _ = self.soft_attention_so(h_tilde_so, enc_out)
        soft_attn_oa, _ = self.soft_attention_oa(h_tilde_oa, enc_out)

        ctx_mas = torch.repeat_interleave(soft_attn_as.unsqueeze(1), full_feature_len, dim=1)
        ctx_mso = torch.repeat_interleave(soft_attn_so.unsqueeze(1), full_feature_len, dim=1)
        ctx_moa = torch.repeat_interleave(soft_attn_oa.unsqueeze(1), full_feature_len, dim=1)

        return ctx_mas, ctx_mso, ctx_moa


class Lang_Vision_Decoder(nn.Module):
    """ Language Conditioned Visual Graph """

    def __init__(self, embedding_dim, hidden_size, dropout_ratio, feature_size):
        super(Lang_Vision_Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.angle_feat_size = args.angle_feat_size
        # self.drop_env = nn.Dropout(p=args.featdropout)
        self.action_layer = nn.Sequential(nn.Linear(self.angle_feat_size, hidden_size, bias=False), nn.Tanh())
        self.action_embedding_layer = nn.Sequential(nn.Linear(self.angle_feat_size, embedding_dim), nn.Tanh())
        self.global_LSTM = nn.LSTMCell(embedding_dim + feature_size, hidden_size)
        self.global_visual_attn_layer = SoftAttention(hidden_size, feature_size)
        self.dropout = nn.Dropout(p=dropout_ratio)
        self.lang_attn_layer = Language_Attention_Cxt(hidden_size)
        self.lang_attn_rel_layer = Language_Attention_Relational_Cxt(hidden_size)

        self.object_raw_dim = 300
        self.full_obj_attn = SoftObjAttention(hidden_size, self.object_raw_dim)
        self.sc = nn.Sequential(nn.Linear(feature_size - self.angle_feat_size, hidden_size, bias=False), nn.Tanh())
        self.oc = nn.Sequential(nn.Linear(self.object_raw_dim, hidden_size, bias=False), nn.Tanh())
        self.ac = nn.Sequential(nn.Linear(hidden_size * 2, hidden_size, bias=False), nn.Tanh())

        self.si = nn.Sequential(nn.Linear(hidden_size, hidden_size, bias=True), nn.Tanh())
        self.oi = nn.Sequential(nn.Linear(hidden_size, hidden_size, bias=True), nn.Tanh())
        self.ai = nn.Sequential(nn.Linear(hidden_size, hidden_size, bias=True), nn.Tanh())

        self.message_as = nn.Sequential(nn.Linear(hidden_size * 2, hidden_size, bias=True), nn.Tanh())
        self.message_so = nn.Sequential(nn.Linear(hidden_size * 2, hidden_size, bias=True), nn.Tanh())
        self.message_oa = nn.Sequential(nn.Linear(hidden_size * 2, hidden_size, bias=True), nn.Tanh())

        self.message_s = nn.Sequential(nn.Linear(hidden_size * 2, hidden_size, bias=False), nn.Tanh())
        self.message_o = nn.Sequential(nn.Linear(hidden_size * 2, hidden_size, bias=False), nn.Tanh())
        self.message_a = nn.Sequential(nn.Linear(hidden_size * 2, hidden_size, bias=False), nn.Tanh())

        self.final_layer = nn.Linear(hidden_size * 3, 1, bias=False)

    def forward(self, action, feature, full_feat, full_obj_feat, prev_h_t, prev_c_t, prev_act_node, enc_out,
                ctx_mask=None):
        ''' candidate visual features '''
        full_img_feat = full_feat[..., :-self.angle_feat_size]
        ''' candidate action features '''
        full_act_feat = full_feat[..., -self.angle_feat_size:]

        # full_obj_feat = self.drop_env(full_obj_feat)

        ''' directional encoding at the previously selected action direction '''
        prev_act_embeds = self.dropout(self.action_embedding_layer(action))

        ''' attended global visual feature '''
        f_g_tilde, _ = self.global_visual_attn_layer(self.dropout(prev_h_t), feature, output_tilde=False)
        global_feat = torch.cat((prev_act_embeds, f_g_tilde), 1)
        h_t, c_t = self.global_LSTM(global_feat, (prev_h_t, prev_c_t))

        ''' Language Attention graph Specialized contexts '''
        h_tilde_G, h_tilde_S, h_tilde_O, h_tilde_A, cxt_s, cxt_o, cxt_a = self.lang_attn_layer(h_t, enc_out, ctx_mask,
                                                                                               full_feat.size(1))
        ''' Language Attention graph relational contexts '''
        ctx_mas, ctx_mso, ctx_moa = self.lang_attn_rel_layer(h_tilde_S, h_tilde_O, h_tilde_A, enc_out,
                                                             full_feat.size(1))

        ''' Node initialization '''
        full_obj_feat, _ = self.full_obj_attn(h_tilde_O, full_obj_feat)
        prev_act_node = torch.repeat_interleave(prev_act_node.unsqueeze(1), full_feat.size(1), dim=1)

        action_feat = self.action_layer(full_act_feat)
        s_tilde_c = self.sc(full_img_feat)
        o_tilde_c = self.oc(full_obj_feat)
        ''' define a temporal link of the graph over the direction node '''
        a_tilde_c = self.ac(torch.cat((prev_act_node, action_feat), dim=2))

        ''' Apply an element-wise product between each projected visual feature and the 
        corresponding specialized contexts'''
        s_tilde_i = self.si(cxt_s * s_tilde_c)
        o_tilde_i = self.oi(cxt_o * o_tilde_c)
        a_tilde_i = self.ai(cxt_a * a_tilde_c)
        # scene_tilde_0 = s_tilde_i.clone()
        # object_tilde_0 = o_tilde_i.clone()
        # action_tilde_0 = a_tilde_i.clone()

        ''' Every node receives messages from the other two nodes. Apply a learnable non-linear projection, 
        with Tanh as the activation function. Apply an element-wise product between each message and the 
        relation context and add the initial node feature'''
        m_as = self.message_as(torch.cat((a_tilde_i, s_tilde_i), dim=2))
        m_os = self.message_so(torch.cat((o_tilde_i, s_tilde_i), dim=2))
        normalize = BertLayerNorm(m_as.size()[1:])
        norm_mas = normalize(m_as)
        norm_mos = normalize(m_os)
        m_s = self.message_s(torch.cat((norm_mos * ctx_mso, norm_mas * ctx_mas), dim=2)) + s_tilde_i

        m_so = self.message_so(torch.cat((s_tilde_i, o_tilde_i), dim=2))
        m_ao = self.message_oa(torch.cat((a_tilde_i, o_tilde_i), dim=2))
        norm_mso = normalize(m_so)
        norm_mao = normalize(m_ao)
        m_o = self.message_o(torch.cat((norm_mso * ctx_mso, norm_mao * ctx_moa), dim=2)) + o_tilde_i

        m_oa = self.message_oa(torch.cat((o_tilde_i, a_tilde_i), dim=2))
        m_sa = self.message_as(torch.cat((s_tilde_i, a_tilde_i), dim=2))
        norm_moa = normalize(m_oa)
        norm_msa = normalize(m_sa)
        m_a = self.message_a(torch.cat((norm_moa * ctx_moa, norm_msa * ctx_mas), dim=2)) + a_tilde_i

        ''' Action prediction'''
        logits = self.final_layer(torch.cat((m_s, m_o, m_a), dim=2)).squeeze()

        return h_t, c_t, logits, h_tilde_G, m_a

class ObjEmbedding(nn.Module):
    def __init__(self, weights_matrix, non_trainable=True):
        super().__init__()
        num_embeddings, embedding_dim = weights_matrix.size()
        self.emb_layer = nn.Embedding(num_embeddings, embedding_dim)
        self.emb_layer.load_state_dict({'weight': weights_matrix})
        if non_trainable:
            self.emb_layer.weight.requires_grad = False

    def forward(self, inputs):
        return self.emb_layer(inputs)


class SoftAttention(nn.Module):
    """Soft Dot Attention.
    Ref: http://www.aclweb.org/anthology/D15-1166
    Adapted from PyTorch OPEN NMT.
    """

    def __init__(self, query_dim, ctx_dim):
        # Initialize layer
        super(SoftAttention, self).__init__()
        self.linear_in = nn.Linear(query_dim, ctx_dim, bias=False)
        self.sm = nn.Softmax()
        self.linear_out = nn.Sequential(nn.Linear(query_dim + ctx_dim, query_dim, bias=False), nn.Tanh())

    def forward(self, h, context, mask=None,
                output_tilde=True, output_prob=True):
        """
        Given the target hidden state h and the context vector c_t, we employ a
        simple concatenation layer to combine the information from both vectors to produce an attentional
        hidden state
        """
        target = self.linear_in(h).unsqueeze(2)  # batch x dim x 1

        # Get attention
        attn = torch.bmm(context, target).squeeze(2)  # batch x seq_len
        logit = attn

        if mask is not None:
            attn.masked_fill_(mask, -float('inf'))
        attn = self.sm(attn)
        attn3 = attn.view(attn.size(0), 1, attn.size(1))  # batch x 1 x seq_len

        weighted_context = torch.bmm(attn3, context).squeeze(1)  # batch x dim
        if not output_prob:
            attn = logit
        if output_tilde:
            h_tilde = torch.cat((weighted_context, h), 1)
            h_tilde = self.linear_out(h_tilde)
            return h_tilde, attn
        else:
            return weighted_context, attn


class SoftObjAttention(nn.Module):
    def __init__(self, query_dim, ctx_dim):
        '''Initialize layer.'''
        super(SoftObjAttention, self).__init__()
        self.linear_in = nn.Linear(query_dim, ctx_dim, bias=False)
        self.sm = nn.Softmax()

    def forward(self, h, context, mask=None):
        target = self.linear_in(h).unsqueeze(2)  # batch x dim x 1

        batch_size = context.size(0);
        max_cand = context.size(1)

        target = torch.repeat_interleave(target, context.size(1), dim=0)
        context = context.view(batch_size * max_cand, context.size(2), context.size(3))

        # Get attention
        attn = torch.bmm(context, target).squeeze(2)  # batch x seq_len
        logit = attn

        if mask is not None:
            # -Inf masking prior to the softmax
            attn.masked_fill_(mask, -float('inf'))
        attn = self.sm(attn)  # There will be a bug here, but it's actually a problem in torch source code.
        attn3 = attn.view(attn.size(0), 1, attn.size(1))  # batch x 1 x seq_len

        weighted_context = torch.bmm(attn3, context).squeeze(1)  # batch x dim
        attn = logit

        weighted_context = weighted_context.view(batch_size, max_cand, context.size(-1))
        attn = attn.view(batch_size, max_cand, attn.size(-1))

        return weighted_context, attn


class BertLayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        Code: https://github.com/huggingface/transformers
        """
        super(BertLayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias
      
        
class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()
        self.state2value = nn.Sequential(
            nn.Linear(args.rnn_dim, args.rnn_dim),
            nn.ReLU(),
            nn.Dropout(args.dropout),
            nn.Linear(args.rnn_dim, 1),
        )

    def forward(self, state):
        return self.state2value(state).squeeze()


class SpeakerEncoder(nn.Module):
    def __init__(self, feature_size, hidden_size, dropout_ratio, bidirectional):
        super().__init__()
        self.num_directions = 2 if bidirectional else 1
        self.hidden_size = hidden_size
        self.num_layers = 1
        self.feature_size = feature_size

        self.lstm = nn.LSTM(feature_size, self.hidden_size // self.num_directions, self.num_layers,
                            batch_first=True, dropout=dropout_ratio, bidirectional=bidirectional)
        self.drop = nn.Dropout(p=dropout_ratio)
        self.drop3 = nn.Dropout(p=args.featdropout)
        self.attention_layer = SoftAttention(self.hidden_size, feature_size)

        self.post_lstm = nn.LSTM(self.hidden_size, self.hidden_size // self.num_directions, self.num_layers,
                                 batch_first=True, dropout=dropout_ratio, bidirectional=bidirectional)

    def forward(self, action_embeds, feature, lengths, already_dropfeat=False):
        """
        :param action_embeds: (batch_size, length, 2052). The feature of the view
        :param feature: (batch_size, length, 36, 2052). The action taken (with the image feature)
        :param lengths: Not used in it
        :return: context with shape (batch_size, length, hidden_size)
        """
        x = action_embeds
        if not already_dropfeat:
            x[..., :-args.angle_feat_size] = self.drop3(
                x[..., :-args.angle_feat_size])  # Do not dropout the spatial features

        # LSTM on the action embed
        ctx, _ = self.lstm(x)
        ctx = self.drop(ctx)

        # Att and Handle with the shape
        batch_size, max_length, _ = ctx.size()
        if not already_dropfeat:
            feature[..., :-args.angle_feat_size] = self.drop3(
                feature[..., :-args.angle_feat_size])  # Dropout the image feature
        x, _ = self.attention_layer(  # Attend to the feature map
            ctx.contiguous().view(-1, self.hidden_size),  # (batch, length, hidden) --> (batch x length, hidden)
            feature.view(batch_size * max_length, -1, self.feature_size),
            # (batch, length, # of images, feature_size) --> (batch x length, # of images, feature_size)
        )
        x = x.view(batch_size, max_length, -1)
        x = self.drop(x)

        # Post LSTM layer
        x, _ = self.post_lstm(x)
        x = self.drop(x)

        return x


class SpeakerDecoder(nn.Module):
    def __init__(self, vocab_size, embedding_size, padding_idx, hidden_size, dropout_ratio):
        super().__init__()
        self.hidden_size = hidden_size
        self.embedding = torch.nn.Embedding(vocab_size, embedding_size, padding_idx)
        self.lstm = nn.LSTM(embedding_size, hidden_size, batch_first=True)
        self.drop = nn.Dropout(dropout_ratio)
        self.attention_layer = SoftAttention(hidden_size, hidden_size)
        self.projection = nn.Linear(hidden_size, vocab_size)
        self.baseline_projection = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(dropout_ratio),
            nn.Linear(128, 1)
        )

    def forward(self, words, ctx, ctx_mask, h0, c0):
        embeds = self.embedding(words)
        embeds = self.drop(embeds)
        x, (h1, c1) = self.lstm(embeds, (h0, c0))

        x = self.drop(x)

        # Get the size
        batchXlength = words.size(0) * words.size(1)
        multiplier = batchXlength // ctx.size(0)  # By using this, it also supports the beam-search

        # Att and Handle with the shape
        # Reshaping x          <the output> --> (b(word)*l(word), r)
        # Expand the ctx from  (b, a, r)    --> (b(word)*l(word), a, r)
        # Expand the ctx_mask  (b, a)       --> (b(word)*l(word), a)
        x, _ = self.attention_layer(
            x.contiguous().view(batchXlength, self.hidden_size),
            ctx.unsqueeze(1).expand(-1, multiplier, -1, -1).contiguous().view(batchXlength, -1, self.hidden_size),
            mask=ctx_mask.unsqueeze(1).expand(-1, multiplier, -1).contiguous().view(batchXlength, -1)
        )
        x = x.view(words.size(0), words.size(1), self.hidden_size)

        # Output the prediction logit
        x = self.drop(x)
        logit = self.projection(x)

        return logit, h1, c1
