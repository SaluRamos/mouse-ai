#modules
from train_utils import load_padded_sequences, GANMonitor, MAX_SEQ_LEN, FEATURE_DIM, COND_DIM
#libs
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
import matplotlib.pyplot as plt
#native libs
import os

LATENT_DIM = 100
BATCH_SIZE = 32
EPOCHS = 50

GENERATOR_LEARNING_RATE:float = 0.0001
DISCRIMINATOR_LEARNING_RATE:float = 0.000005

AABB_DISTANCE_LOSS_MULT:float = 1000.0
AABB_FIXED_LOSS:float = 1.0

DISCRIMINATOR_TRAIN_RATE:int = 1
DISCRIMINATOR_DROPOUT:float = 0.2

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
    model = models.Model([noise_input, cond_input], output, name="Generator")
    model.epoch_tracker = model.add_weight(
        name="epoch_tracker",
        shape=(),
        initializer="zeros",
        trainable=False,
        dtype=tf.int32
    )
    return model

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
    x = layers.Dropout(DISCRIMINATOR_DROPOUT)(x)
    x = layers.GlobalMaxPooling1D()(x)
    x = layers.Dense(64, activation="relu")(x)
    output = layers.Dense(1, activation="sigmoid")(x) # 0 a 1
    model = models.Model([seq_input, cond_input], output, name="Discriminator")
    model.epoch_tracker = model.add_weight(
        name="epoch_tracker",
        shape=(),
        initializer="zeros",
        trainable=False,
        dtype=tf.int32
    )
    return model

class MouseGAN(models.Model):

    def __init__(self, generator, discriminator):
        super(MouseGAN, self).__init__()
        self.generator = generator
        self.discriminator = discriminator
        if not hasattr(self.generator, 'epoch_tracker'):
             self.generator.epoch_tracker = self.generator.add_weight(
                name="epoch_tracker",
                shape=(),
                initializer="zeros",
                trainable=False,
                dtype=tf.int32
            )
        if not hasattr(self.discriminator, 'epoch_tracker'):
             self.discriminator.epoch_tracker = self.discriminator.add_weight(
                name="epoch_tracker",
                shape=(),
                initializer="zeros",
                trainable=False,
                dtype=tf.int32
            )

    def compile(self, d_optimizer, g_optimizer, loss_fn, **kwargs):
        super(MouseGAN, self).compile(**kwargs)
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.loss_fn = loss_fn
        self.d_loss_metric = tf.keras.metrics.Mean(name="d_loss")
        self.g_loss_metric = tf.keras.metrics.Mean(name="g_loss")
        self.train_step_counter = tf.Variable(0, trainable=False, dtype=tf.int32)

    @property
    def metrics(self):
        return [self.d_loss_metric, self.g_loss_metric]

    def train_step(self, data):
        real_seqs, conditions = data
        real_seqs = tf.cast(real_seqs, tf.float32)
        conditions = tf.cast(conditions, tf.float32)
        batch_size = tf.shape(real_seqs)[0]

        self.train_step_counter.assign_add(1)
        train_factor = tf.cast(tf.equal(tf.math.mod(self.train_step_counter, DISCRIMINATOR_TRAIN_RATE), 0), tf.float32)

        random_latent_vectors = tf.random.normal(shape=(batch_size, LATENT_DIM))
        generated_seqs = self.generator([random_latent_vectors, conditions])

        labels_real = tf.ones((batch_size, 1)) * 0.9
        labels_fake = tf.zeros((batch_size, 1))

        # labels_real = tf.random.uniform(shape=(batch_size, 1), minval=0.8, maxval=1.0)
        # labels_fake = tf.random.uniform(shape=(batch_size, 1), minval=0.0, maxval=0.2)

        with tf.GradientTape() as tape:
            pred_real = self.discriminator([real_seqs, conditions])
            pred_fake = self.discriminator([generated_seqs, conditions])
            d_loss_real = self.loss_fn(labels_real, pred_real)
            d_loss_fake = self.loss_fn(labels_fake, pred_fake)
            d_loss = d_loss_real + d_loss_fake

        acc_real = tf.reduce_mean(tf.cast(pred_real > 0.5, tf.float32))
        acc_fake = tf.reduce_mean(tf.cast(pred_fake <= 0.5, tf.float32))

        grads = tape.gradient(d_loss, self.discriminator.trainable_weights)
        d_grads = [g * train_factor for g in grads]
        self.d_optimizer.apply_gradients(zip(d_grads, self.discriminator.trainable_weights))

        # --- 2. Treinar Gerador ---
        misleading_labels = tf.ones((batch_size, 1))

        with tf.GradientTape() as tape:
            # Gerar sequências dentro do tape
            generated_seqs = self.generator([random_latent_vectors, conditions])
            pred_fake = self.discriminator([generated_seqs, conditions])
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
            dist_total = dist_x + dist_y
            penalty_mask = tf.cast(tf.greater(dist_total, 0), tf.float32) #retorna 0 ou 1
            aabb_distance_loss = tf.reduce_mean(dist_total + (penalty_mask * AABB_FIXED_LOSS))
            g_loss = g_gan_loss + (aabb_distance_loss * AABB_DISTANCE_LOSS_MULT)
            #calcular taxa de acerto do gerador em terminar dentro do botão
            within_x = tf.logical_and(px >= x_min, px <= x_max)
            within_y = tf.logical_and(py >= y_min, py <= y_max)
            is_hit = tf.logical_and(within_x, within_y)
            g_hit_rate = tf.reduce_mean(tf.cast(is_hit, tf.float32))

        grads = tape.gradient(g_loss, self.generator.trainable_weights)
        self.g_optimizer.apply_gradients(zip(grads, self.generator.trainable_weights))

        # Atualiza métricas
        self.d_loss_metric.update_state(d_loss)
        self.g_loss_metric.update_state(g_loss)
        
        return {
            "d_loss": self.d_loss_metric.result(),
            "g_loss": self.g_loss_metric.result(),
            "aabb_distance_loss": aabb_distance_loss,
            "acc_real": acc_real,
            "acc_fake": acc_fake,
            "g_hit": g_hit_rate
        }

if __name__ == "__main__":
    X_seq, X_cond = load_padded_sequences()
    generator = None
    discriminator = None

    generator_save_path = "models/mouse_gan_generator.keras"
    discriminator_save_path = "models/mouse_gan_discriminator.keras"
    if not os.path.exists("models/"):
        generator_save_path = "../" + generator_save_path
        discriminator_save_path = "../" + discriminator_save_path

    try:
        generator = models.load_model(generator_save_path)
    except Exception as e:
        generator = build_generator()
    try:
        discriminator = models.load_model(discriminator_save_path)
    except Exception as e:
        discriminator = build_discriminator()

    print(f"discriminator: {discriminator.epoch_tracker}")
    print(f"generator: {generator.epoch_tracker}")

    # print("\n--- Generator Summary ---")
    # generator.summary()
    # print("\n--- Discriminator Summary ---")
    # discriminator.summary()

    gan = MouseGAN(generator, discriminator)
    
    gan.compile(
        d_optimizer=optimizers.Adam(learning_rate=DISCRIMINATOR_LEARNING_RATE),
        g_optimizer=optimizers.Adam(learning_rate=GENERATOR_LEARNING_RATE),
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

    generator_new_epoch_value = generator.epoch_tracker.read_value() + EPOCHS
    discriminator_new_epoch_value = discriminator.epoch_tracker.read_value() + EPOCHS
    generator.epoch_tracker.assign(generator_new_epoch_value)
    discriminator.epoch_tracker.assign(discriminator_new_epoch_value)

    generator.save(generator_save_path)
    discriminator.save(discriminator_save_path)

    plt.figure(figsize=(10, 5))
    plt.plot(history.history['acc_real'], label='taxa de acerto do discriminador')
    plt.plot(history.history['acc_fake'], label='taxa de erro do discriminador')
    plt.plot(history.history['g_hit'], label='taxa de hit do gerador')
    plt.title('GAN Training Progress')
    plt.xlabel('Epoch')
    plt.ylabel('%')
    plt.legend()
    plt.grid(True)
    plt.show()