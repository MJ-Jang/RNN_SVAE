################################################## 1. Import modules ###################################################
import gensim
import code.helper as hp
from code.utils import *
import tensorflow as tf
import os
import time


########################################################################################################################

class RAE:
    def __init__(self, word2vec_path, hidden_dim, latent_dim):

        # word properties
        self.word2vec_model = gensim.models.Word2Vec.load(word2vec_path)
        self.vocab_size = len(self.word2vec_model.wv.index2word) + 1
        self.word_vec_dim = self.word2vec_model.vector_size
        self.lookup = [[0.] * self.word_vec_dim] + [x for x in self.word2vec_model.wv.syn0]

        self.PAD = 0
        self.EOS = self.word2vec_model.wv.vocab['EOS'].index + 1

        # model hyper-parameters
        self.input_embedding_size = self.word_vec_dim
        self.encoder_hidden_units = hidden_dim
        self.latent_units = latent_dim
        self.decoder_hidden_units = self.latent_units

        # placeholders
        # input, target shape : [max_time, batch_size]
        # sequence length shape : [batch_size]
        self.encoder_inputs = tf.placeholder(shape=(None, None), dtype=tf.int32, name='encoder_inputs')
        self.decoder_targets = tf.placeholder(shape=(None, None), dtype=tf.int32, name='decoder_targets')
        self.decoder_inputs = tf.placeholder(shape=(None, None), dtype=tf.int32, name='decoder_inputs')
        self.encoder_sequence_len = tf.placeholder(shape=(None,), dtype=tf.int32, name='en_sequence_len')
        self.decoder_sequence_len = tf.placeholder(shape=(None,), dtype=tf.int32, name='de_sequence_len')

        self.learning_rate = tf.placeholder(shape=(), dtype=tf.float32, name="learning_rate")

        # Embedding matrix
        # denote lookup table as placeholder
        self.embeddings = tf.placeholder(shape=(self.vocab_size, self.input_embedding_size), dtype=tf.float32)
        self.encoder_inputs_embedded = tf.nn.embedding_lookup(self.embeddings, self.encoder_inputs)
        self.decoder_inputs_embedded = tf.nn.embedding_lookup(self.embeddings, self.decoder_inputs)

    def params(self):
        self.weights = {
            'mu_w': tf.get_variable("mu_w", shape=[self.input_embedding_size + self.encoder_hidden_units * 2, self.latent_units],
                                    initializer=tf.contrib.layers.xavier_initializer()),
            'sd_w': tf.get_variable("sd_w", shape=[self.input_embedding_size + self.encoder_hidden_units * 2, self.latent_units],
                                    initializer=tf.contrib.layers.xavier_initializer()),
        }

        self.bias = {
            'mu_b': tf.Variable(tf.random_normal([self.latent_units], stddev=0.1), name="mu_b"),
            'sd_b': tf.Variable(tf.random_normal([self.latent_units], stddev=0.1), name="sd_b"),
        }

    # Encoder
    def encoder(self, name):
        # sentence encoder
        with tf.variable_scope("encoder"):
            self.encoder_cell_fw = tf.contrib.rnn.GRUCell(self.encoder_hidden_units)
            self.encoder_cell_bw = tf.contrib.rnn.GRUCell(self.encoder_hidden_units)

            # semantic vector is final state of encoder
            self.encoder_output, self.encoder_final_state = tf.nn.bidirectional_dynamic_rnn(self.encoder_cell_fw, self.encoder_cell_bw,
                                                                          inputs=self.encoder_inputs_embedded,
                                                                          sequence_length=self.encoder_sequence_len,
                                                                          dtype=tf.float32, time_major=True, scope=name)
            self.final_concat = tf.concat(self.encoder_final_state, axis=1)

            # word weight
            self.align_fw = tf.reduce_sum(tf.multiply(self.encoder_output[0], self.encoder_final_state[0]), 2)
            self.align_bw = tf.reduce_sum(tf.multiply(self.encoder_output[1], self.encoder_final_state[1]), 2)

            self.alpha_fw = tf.nn.softmax(self.align_fw, 0)
            self.alpha_bw = tf.nn.softmax(self.align_bw, 0)

            self.alpha_total = (self.alpha_fw + self.alpha_bw) / 2

            # calculate semantic vector
            mul_alpha = tf.multiply(tf.transpose(self.alpha_total, [1, 0]), tf.transpose(self.encoder_inputs_embedded, [2, 1, 0]))
            mul_alpha = tf.transpose(mul_alpha, [2, 1, 0])
            self.semantic = tf.reduce_sum(mul_alpha, 0)

            self.final_state = tf.concat([self.final_concat, self.semantic], axis=1)

        # Mu encoder
        with tf.variable_scope("enc_mu"):
            self.enc_mu = tf.matmul(self.final_state, self.weights['mu_w']) + self.bias['mu_b']
        # Sigma encoder
        with tf.variable_scope("enc_sd"):
            self.enc_sd = tf.matmul(self.final_state, self.weights['sd_w']) + self.bias['sd_b']

        # Sample epsilon
        epsilon = tf.random_normal(tf.shape(self.enc_sd), name='epsilon')

        # Sample latent variable
        std_encoder = tf.exp(.5 * self.enc_sd)

        # Compute KL divergence (latent loss)
        self.KLD = -.5 * tf.reduce_sum(
            1. - tf.square(self.enc_sd) - tf.square(self.enc_mu) + tf.log(tf.square(self.enc_sd) + 1e-8), 1)

        # Generate z(latent)
        # z = mu + (sigma * epsilon)
        self.z = self.enc_mu + tf.multiply(std_encoder, epsilon)

    # Decoder
    def decoder(self, name):
        with tf.variable_scope("decoder"):
            self.decoder_cell = tf.contrib.rnn.GRUCell(self.decoder_hidden_units)

            self.decoder_outputs, self.decoder_final_state = tf.nn.dynamic_rnn(self.decoder_cell,
                                                                               self.decoder_inputs_embedded,
                                                                               sequence_length=self.decoder_sequence_len,
                                                                               initial_state=self.z, dtype=tf.float32,
                                                                               time_major=True, scope=name, )

    def model(self, softmax_sampling_size, softmax_name, bias_name):
        self.decoder_softmax_weight = tf.get_variable(softmax_name, shape=[self.vocab_size, self.decoder_hidden_units],
                                                      initializer=tf.contrib.layers.xavier_initializer())
        self.decoder_softmax_bias = tf.Variable(tf.random_normal([self.vocab_size], stddev=0.1), name=bias_name)

        # sampling softmax cross entropy loss
        # make batch to flat for easy calculation
        self.sampled_softmax_cross_entropy_loss = tf.nn.sampled_softmax_loss(weights=self.decoder_softmax_weight,
                                                                             biases=self.decoder_softmax_bias,
                                                                             labels=tf.reshape(self.decoder_targets,
                                                                                               [-1, 1]),
                                                                             inputs=tf.reshape(self.decoder_outputs,
                                                                                               [-1, self.decoder_hidden_units]),
                                                                             num_sampled=softmax_sampling_size,
                                                                             num_classes=self.vocab_size, num_true=1)
        self.BCE = tf.reduce_sum(self.sampled_softmax_cross_entropy_loss)
        self.total_loss = tf.reduce_mean(self.KLD + self.BCE)

        self.train_op = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.total_loss)

    def log_and_saver(self, log_path, model_path, sess):
        # log
        self.loss_sum = tf.summary.scalar("Loss", self.total_loss)
        self.summary = tf.summary.merge_all()

        self.writer_tr = tf.summary.FileWriter(log_path + "/train", sess.graph)
        self.writer_test = tf.summary.FileWriter(log_path + "/test", sess.graph)

        # saver
        self.dir = os.path.dirname(os.path.realpath(model_path))

    def saver(self):
        self.all_saver = tf.train.Saver()

    # feed_dict function
    def next_feed(self, batch, learning_rate):
        self.encoder_inputs_, self.en_seq_len_ = hp.batch(batch)
        self.decoder_targets_, self.de_seq_len_ = hp.batch([(sequence) + [self.EOS] for sequence in batch])
        self.decoder_inputs_, _ = hp.batch([[self.EOS] + (sequence) for sequence in batch])

        return {
            self.encoder_inputs: self.encoder_inputs_,
            self.decoder_inputs: self.decoder_inputs_,
            self.decoder_targets: self.decoder_targets_,
            self.embeddings: self.lookup,
            self.encoder_sequence_len: self.en_seq_len_,
            self.decoder_sequence_len: self.de_seq_len_,
            self.learning_rate: learning_rate,
        }

    def variable_initialize(self, sess):
        sess.run(tf.global_variables_initializer())

    def train(self, corpus_train, corpus_val, batch_size, n_epoch, init_lr, sess):

        print("Start train !!!!!!!")
        count_t = time.time()

        for i in range(n_epoch):

            if (i+1) <= 10:
                lr = init_lr
            else:
                lr = init_lr * 0.1

            for start, end in zip(range(0, len(corpus_train), batch_size),
                                  range(batch_size, len(corpus_train), batch_size)):

                batch_time_50 = time.time()

                global_step = i * int(len(corpus_train) / batch_size) + int(start / batch_size + 1)

                ## training
                fd = self.next_feed(corpus_train[start:end], learning_rate=lr)
                s_tr, _, l_tr = sess.run([self.summary, self.train_op, self.total_loss], feed_dict=fd)
                self.writer_tr.add_summary(s_tr, global_step)

                # validation
                tst_idx = np.arange(len(corpus_val))
                np.random.shuffle(tst_idx)
                tst_idx = tst_idx[0:batch_size]


                fd_tst = self.next_feed(np.take(corpus_val, tst_idx, 0), learning_rate=lr)


                s_tst, l_tst = sess.run([self.summary, self.total_loss], feed_dict=fd_tst)
                self.writer_test.add_summary(s_tst, global_step)

                if start == 0 or int(start / batch_size + 1) % 50 == 0:
                    print("Iter", int(start / batch_size + 1), " Training Loss:", l_tr, "Test loss : ", l_tst,
                          "Time : ", time.time() - batch_time_50)

            if (i + 1) % 10 == 0:
                savename = self.dir + "net-" + str(i + 1) + ".ckpt"
                self.all_saver.save(sess=sess, save_path=savename)

            print("epoch : ", i + 1, "loss : ", l_tr, "Test loss : ", l_tst)

        print("Running Time : ", time.time() - count_t)
        print("Training Finished!!!")

    def load_model(self, model_path, model_name, sess):
        restorename = model_path + "/" + model_name
        self.all_saver.restore(sess, restorename)

    # semantic calculation
    # Auto encoder models should be loaded first
    def calculate_dist(self,corpus,batch_size, sampling_num,sess):
        for start, end in zip(range(0, len(corpus), batch_size), range(batch_size, len(corpus), batch_size)):
            encoder_final = []
            for i in range(sampling_num):
                 tmp = sess.run(self.z, feed_dict=self.next_feed(corpus[start:end], learning_rate=0.01))
                 encoder_final.append(tmp)

            encoder_final = np.mean(encoder_final,0)

            if start == 0:
                output = encoder_final

            else:
                output = np.vstack((output, encoder_final))

            print("Finished to", start,"th documents!!!!")

        return output

    # Beam Search Part
    def Beamsearch_options(self, beam_size, max_len, sampling_num):

        self.beam_size = beam_size
        self.max_len = max_len
        self.beam_input = tf.placeholder(shape=(None, None), dtype=tf.int32, name='beam_input')
        self.beam_input_embedded = tf.nn.embedding_lookup(self.embeddings, self.beam_input)

        self.final_plc = tf.placeholder(tf.float32, [None, 2 * self.encoder_hidden_units  +  self.input_embedding_size])
        self.state_plc = tf.placeholder(tf.float32, [None, self.decoder_hidden_units])

        enc_mu = tf.matmul(self.final_plc, self.weights['mu_w']) + self.bias['mu_b']
        enc_sd = tf.matmul(self.final_plc, self.weights['sd_w']) + self.bias['sd_b']

        mean_latent = []
        for i in range(sampling_num):
            # computations for beam
            epsilon = tf.random_normal(tf.shape(enc_sd))
            # Sample latent variable
            std_encoder = tf.exp(.5 * enc_sd)

            # Generate z(latent)
            mean_latent.append(enc_mu + tf.multiply(std_encoder, epsilon))

        self.mean_latent = tf.reduce_mean(mean_latent, 0)

        self.beam_outputs_init, self.beam_state_init = tf.nn.dynamic_rnn(self.decoder_cell, self.beam_input_embedded,
                                                                         initial_state=self.mean_latent,
                                                                         dtype=tf.float32, time_major=True,
                                                                         scope="beam_decoder")

        self.beam_outputs_init = tf.reshape(self.beam_outputs_init, [-1, self.decoder_hidden_units])
        self.logits_init = tf.matmul(self.beam_outputs_init,
                                     tf.transpose(self.decoder_softmax_weight)) + self.decoder_softmax_bias
        self.prob_pred_init, self.word_pred_init = tf.nn.top_k(tf.nn.softmax(self.logits_init), k=self.beam_size,
                                                               sorted=False)

        # after initial state
        self.beam_outputs, self.beam_state = tf.nn.dynamic_rnn(self.decoder_cell, self.beam_input_embedded,
                                                               initial_state=self.state_plc,
                                                               dtype=tf.float32, time_major=True,
                                                               scope="beam_decoder")

        self.beam_outputs = tf.reshape(self.beam_outputs, [-1, self.decoder_hidden_units])
        self.logits = tf.matmul(self.beam_outputs,
                                tf.transpose(self.decoder_softmax_weight)) + self.decoder_softmax_bias
        self.prob_pred, self.word_pred = tf.nn.top_k(tf.nn.softmax(self.logits), k=self.beam_size, sorted=False)

    def next_feed_beam_init(self, word_input, final_input):
        beam_input_, _ = hp.batch(word_input)

        return {
            self.embeddings: self.lookup,
            self.beam_input: beam_input_,
            self.final_plc: final_input
        }

    def next_feed_beam(self, word_input, state_input):
        beam_input_, _ = hp.batch(word_input)

        return {
            self.embeddings: self.lookup,
            self.beam_input: beam_input_,
            self.state_plc: state_input
        }

    def chose_highscores(self, score_mat):
        flat_mat = np.ndarray.flatten(score_mat)
        ix = flat_mat.argsort()[-self.beam_size:][::-1]

        for i in range(len(flat_mat)):
            if i in ix:
                flat_mat[i] = 1
            else:
                flat_mat[i] = 0

        return flat_mat

    def BeamSearchDecoder(self, final_state, sess):

        ## model start

        dead_sample = []
        dead_score = []
        dead_k = len(dead_sample)

        l = 0

        while dead_k == 0 and l < self.max_len:

            if l == 0:
                # initial words : EOS token
                init_words = np.array([[self.EOS]] * self.beam_size)

                encoder_final = np.concatenate([final_state for l in range(self.beam_size)], axis=0)

                # decoding
                fd = self.next_feed_beam_init(word_input=init_words, final_input=encoder_final)
                w, s, state = sess.run([self.word_pred_init, self.prob_pred_init, self.beam_state_init], feed_dict=fd)

                # update live sample, score
                live_sample = np.array([[w[0][i]] for i in range(self.beam_size)])
                live_scores = np.array([[np.log(s[0][i])] for i in range(self.beam_size)])

            else:

                # Search
                # beam size = batch size
                iter_words = np.array([[w] for w in live_sample[:, -1]])

                # decoding
                fd = self.next_feed_beam(word_input=iter_words, state_input=state)

                w, s, state = sess.run([self.word_pred, self.prob_pred, self.beam_state], feed_dict=fd)

                # calculate candidate score
                cand_scores = live_scores + np.log(s)
                cand_scores_flat = np.ndarray.flatten(cand_scores)

                # find candidate word
                cand_sample = np.array([[live_sample[i]] * self.beam_size for i in range(self.beam_size)])
                cand_sample = cand_sample.reshape([self.beam_size * self.beam_size, -1])

                cand_words = np.array([[w[i, j]] for i in range(self.beam_size) for j in range(self.beam_size)])
                cand_words_list = np.concatenate((cand_sample, cand_words), axis=1)

                # find top beam_size word get mask matrix
                selected_idx = self.chose_highscores(cand_scores).astype(int)
                mask = selected_idx > 0

                # select live score and live sample
                live_scores = np.array([s for s, m in zip(cand_scores_flat, mask) if m]).reshape([self.beam_size, 1])
                live_sample = np.array([s for s, m in zip(cand_words_list, mask) if m]).reshape([self.beam_size, -1])

                # find zombies (needs to die)
                zombie = [s[-1] == self.EOS or len(s) >= self.max_len for s in live_sample]

                # add zombies to the dead
                dead_sample += [s for s, z in zip(live_sample, zombie) if z]
                dead_score += [s for s, z in zip(live_scores, zombie) if z]
                dead_k = len(dead_score)

            l += 1

        idx = np.argmax(dead_score)
        answer = dead_sample[idx]

        # remove last EOS token
        if answer[-1] == self.EOS:
            answer = np.delete(answer, -1)

        return answer
