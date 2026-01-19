#modules
from train_utils import load_padded_sequences, GANMonitor, MAX_SEQ_LEN, FEATURE_DIM, COND_DIM
#libs
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
import matplotlib.pyplot as plt
#native libs
import os

LATENT_DIM = 100
BATCH_SIZE = 64
EPOCHS = 10
CONTINUE_TRAINING = True

def build_generator() -> models.Model:
    """
    Entrada: Ruído (Latent) + Condição (Distância inicial, tamanho botão)
    Saída: Sequência de movimentos (MAX_SEQ_LEN, 2)
    """
    noise_input = layers.Input(shape=(LATENT_DIM,), name="noise_input")
    cond_input = layers.Input(shape=(COND_DIM,), name="cond_input")
    x = layers.Concatenate()([noise_input, cond_input])
    x = layers.RepeatVector(MAX_SEQ_LEN)(x)
    x = layers.LSTM(128, return_sequences=True)(x)
    x = layers.LSTM(64, return_sequences=True)(x)
    output = layers.TimeDistributed(layers.Dense(FEATURE_DIM, activation="linear"))(x)
    return models.Model([noise_input, cond_input], output, name="Generator")

def build_discriminator() -> models.Model:
    """
    Entrada: Sequência de movimentos + Condição
    Saída: Probabilidade de ser Real (1.0) ou Fake (0.0)
    """
    seq_input = layers.Input(shape=(MAX_SEQ_LEN, FEATURE_DIM), name="seq_input")
    cond_input = layers.Input(shape=(COND_DIM,), name="cond_input")
    cond_repeated = layers.RepeatVector(MAX_SEQ_LEN)(cond_input)
    x = layers.Concatenate()([seq_input, cond_repeated])
    x = layers.LSTM(64, return_sequences=True)(x)
    x = layers.Dropout(0.1)(x)
    x = layers.GlobalMaxPooling1D()(x) # Pega as características mais fortes
    x = layers.Dense(64, activation="relu")(x)
    output = layers.Dense(1, activation="sigmoid")(x) # 0 a 1
    return models.Model([seq_input, cond_input], output, name="Discriminator")

class MouseGAN(models.Model):

    def __init__(self, generator, discriminator):
        super(MouseGAN, self).__init__()
        self.generator = generator
        self.discriminator = discriminator

    def compile(self, d_optimizer, g_optimizer, loss_fn, **kwargs):
        super(MouseGAN, self).compile(**kwargs)
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.loss_fn = loss_fn
        self.d_loss_metric = tf.keras.metrics.Mean(name="d_loss")
        self.g_loss_metric = tf.keras.metrics.Mean(name="g_loss")

    @property
    def metrics(self):
        return [self.d_loss_metric, self.g_loss_metric]

    def train_step(self, data):
        # Desempacota dados reais
        # O fit(x, y) envia uma tupla (x, y). Se houver erro aqui, os dados não foram passados corretamente.
        real_seqs, conditions = data
        # GARANTIA DE TIPOS: Força tudo para float32 para evitar conflitos de Double vs Float
        real_seqs = tf.cast(real_seqs, tf.float32)
        conditions = tf.cast(conditions, tf.float32)
        batch_size = tf.shape(real_seqs)[0]
        # --- 1. Treinar Discriminador ---
        random_latent_vectors = tf.random.normal(shape=(batch_size, LATENT_DIM))
        # Gera sequências falsas
        generated_seqs = self.generator([random_latent_vectors, conditions])

        labels_real = tf.ones((batch_size, 1)) * 0.9
        # labels_real = tf.random.uniform(shape=(batch_size, 1), minval=0.8, maxval=1.0)
        labels_fake = tf.zeros((batch_size, 1))
        # labels_fake = tf.random.uniform(shape=(batch_size, 1), minval=0.0, maxval=0.2)

        # Combinamos tudo para um único passo de gradiente (opcional, mas às vezes mais estável)
        # Mas vamos manter separado como no original para clareza
        with tf.GradientTape() as tape:
            pred_real = self.discriminator([real_seqs, conditions])
            pred_fake = self.discriminator([generated_seqs, conditions])
            d_loss_real = self.loss_fn(labels_real, pred_real)
            d_loss_fake = self.loss_fn(labels_fake, pred_fake)
            d_loss = d_loss_real + d_loss_fake

        grads = tape.gradient(d_loss, self.discriminator.trainable_weights)
        self.d_optimizer.apply_gradients(zip(grads, self.discriminator.trainable_weights))

        # --- 2. Treinar Gerador ---
        misleading_labels = tf.ones((batch_size, 1))

        with tf.GradientTape() as tape:
            # Gerar sequências dentro do tape
            generated_seqs = self.generator([random_latent_vectors, conditions])
            pred_fake = self.discriminator([generated_seqs, conditions])
            # Perda da GAN (parecer humano)
            g_gan_loss = self.loss_fn(misleading_labels, pred_fake)
            # Adiciona a loss do gerador a distancia do menor caminho até o botão
            path_final_position = tf.reduce_sum(generated_seqs[:, :, :2], axis=1)
            px = path_final_position[:, 0]
            py = path_final_position[:, 1]
            target_x = conditions[:, 0]
            target_y = conditions[:, 1]
            bw = conditions[:, 2]
            bh = conditions[:, 3]
            x_min = target_x - (bw / 2)
            x_max = target_x + (bw / 2)
            y_min = target_y - (bh / 2)
            y_max = target_y + (bh / 2)
            closest_x = tf.clip_by_value(px, x_min, x_max)
            closest_y = tf.clip_by_value(py, y_min, y_max)
            dist_x = tf.abs(px - closest_x)
            dist_y = tf.abs(py - closest_y)
            aabb_distance_loss = tf.reduce_mean(dist_x + dist_y)
            g_loss = g_gan_loss + (aabb_distance_loss * 15.0)

        grads = tape.gradient(g_loss, self.generator.trainable_weights)
        self.g_optimizer.apply_gradients(zip(grads, self.generator.trainable_weights))

        # Atualiza métricas
        self.d_loss_metric.update_state(d_loss)
        self.g_loss_metric.update_state(g_loss)
        
        return {
            "d_loss": self.d_loss_metric.result(),
            "g_loss": self.g_loss_metric.result()
        }

if __name__ == "__main__":
    X_seq, X_cond = load_padded_sequences()
    generator = None
    discriminator = None    
    if CONTINUE_TRAINING:
        generator = models.load_model("../models/mouse_gan_generator.keras")
        discriminator = models.load_model("../models/mouse_gan_discriminator.keras")
    else:
        generator = build_generator()
        discriminator = build_discriminator()
    print("\n--- Generator Summary ---")
    generator.summary()
    print("\n--- Discriminator Summary ---")
    discriminator.summary()

    gan = MouseGAN(generator, discriminator)
    
    gan.compile(
        d_optimizer=optimizers.Adam(learning_rate=0.00003),
        g_optimizer=optimizers.Adam(learning_rate=0.0001), # Gerador geralmente aprende mais devagar
        loss_fn=tf.keras.losses.BinaryCrossentropy()
    )

    print(f"X_seq shape = {X_seq.shape}")
    print(f"X_cond shape = {X_cond.shape}")

    history = gan.fit(
        x=X_seq, y=X_cond,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        callbacks=[GANMonitor(LATENT_DIM)]
    )

    # Salvar Modelo Gerador (O discriminador não é necessário para inferência)
    generator_save_path = "models/mouse_gan_generator.keras"
    discriminator_save_path = "models/mouse_gan_discriminator.keras"
    if not os.path.exists("models/"):
        generator_save_path = "../" + generator_save_path
        discriminator_save_path = "../" + discriminator_save_path
    generator.save(generator_save_path)
    discriminator.save(discriminator_save_path)

    plt.figure(figsize=(10, 5))
    plt.plot(history.history['d_loss'], label='Discriminator Loss')
    plt.plot(history.history['g_loss'], label='Generator Loss')
    plt.title('GAN Training Progress')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()