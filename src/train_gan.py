#modules
from train_utils import load_padded_sequences, GANMonitor, MAX_SEQ_LEN, FEATURE_DIM, COND_DIM
from utils import get_base_path, get_best_model
#libs
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
import matplotlib.pyplot as plt

EPOCH_TO_LOAD = 1298
LOAD_BEST = False
LOAD_DISCRIMINATOR = False

LATENT_DIM = 100
BATCH_SIZE = 32
EPOCHS = 10000

GENERATOR_LEARNING_RATE:float = 0.0001
DISCRIMINATOR_LEARNING_RATE:float = 0.00001

AABB_DISTANCE_LOSS_MULT:float = 1.0
AABB_FIXED_LOSS:float = 1.0

MAX_ACCURACY = 0.8
DISCRIMINATOR_DROPOUT:float = 0.2

def build_generator() -> models.Model:
    """
    Entrada: Ruído (Latent) + Condição (Distância inicial, tamanho botão)
    Saída: Sequência de movimentos (MAX_SEQ_LEN, 2)
    """
    noise_input = layers.Input(shape=(LATENT_DIM,), name="noise_input")
    cond_input = layers.Input(shape=(COND_DIM,), name="cond_input")

    # x = layers.Concatenate()([noise_input, cond_input])
    # x = layers.RepeatVector(MAX_SEQ_LEN)(x)
    # x = layers.LSTM(128, return_sequences=True)(x)
    # x = layers.LSTM(64, return_sequences=True)(x)
    # output = layers.TimeDistributed(layers.Dense(FEATURE_DIM, activation="linear"))(x)
    # model = models.Model([noise_input, cond_input], output, name="Generator")

    # combined = layers.Concatenate()([noise_input, cond_input])
    # x = layers.Dense(256, activation="leaky_relu")(combined)
    # x = layers.BatchNormalization()(x)
    # x = layers.Dense(512, activation="leaky_relu")(x)
    # x = layers.RepeatVector(MAX_SEQ_LEN)(x)
    # x = layers.Bidirectional(layers.LSTM(256, return_sequences=True))(x)
    # x = layers.BatchNormalization()(x)
    # x = layers.LSTM(128, return_sequences=True)(x)
    # x = layers.LSTM(64, return_sequences=True)(x)
    # x = layers.TimeDistributed(layers.Dense(64, activation="leaky_relu"))(x)
    # output = layers.TimeDistributed(layers.Dense(FEATURE_DIM, activation="linear"))(x)
    # model = models.Model([noise_input, cond_input], output, name="Generator_Pro")

    x = layers.Concatenate()([noise_input, cond_input])
    x = layers.Dense(256)(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(512)(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    x = layers.BatchNormalization()(x)
    total_output_size = MAX_SEQ_LEN * FEATURE_DIM
    x = layers.Dense(total_output_size, activation="linear")(x)
    output = layers.Reshape((MAX_SEQ_LEN, FEATURE_DIM), name="seq_output")(x)
    model = models.Model([noise_input, cond_input], output, name="MLP_Generator")

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
    return model

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
        self.train_step_counter = tf.Variable(0, trainable=False, dtype=tf.int32)

    @property
    def metrics(self):
        return [self.d_loss_metric, self.g_loss_metric]

    def train_step(self, data):
        real_seqs, conditions = data
        real_seqs = tf.cast(real_seqs, tf.float32)
        conditions = tf.cast(conditions, tf.float32)
        batch_size = tf.shape(real_seqs)[0]

        random_latent_vectors = tf.random.normal(shape=(batch_size, LATENT_DIM))
        generated_seqs = self.generator([random_latent_vectors, conditions])

        pred_real_pre = self.discriminator([real_seqs, conditions], training=False)
        pred_fake_pre = self.discriminator([generated_seqs, conditions], training=False)
        acc_real_pre = tf.reduce_mean(tf.cast(pred_real_pre > 0.5, tf.float32))
        acc_fake_pre = tf.reduce_mean(tf.cast(pred_fake_pre <= 0.5, tf.float32))
        avg_acc = (acc_real_pre + acc_fake_pre) / 2.0
        d_train_condition = tf.cast(tf.less(avg_acc, MAX_ACCURACY), tf.float32)

        # dynamic_smoothing = 0.9 - (acc_real_pre * 0.2) 
        # labels_real = tf.ones((batch_size, 1)) * dynamic_smoothing
        labels_real = tf.ones((batch_size, 1)) * 0.9
        labels_fake = tf.zeros((batch_size, 1)) * 0.1

        with tf.GradientTape() as tape:
            pred_real = self.discriminator([real_seqs, conditions])
            pred_fake = self.discriminator([generated_seqs, conditions])
            d_loss_real = self.loss_fn(labels_real, pred_real)
            d_loss_fake = self.loss_fn(labels_fake, pred_fake)
            d_loss = d_loss_real + d_loss_fake

        acc_real = tf.reduce_mean(tf.cast(pred_real > 0.5, tf.float32))
        acc_fake = tf.reduce_mean(tf.cast(pred_fake <= 0.5, tf.float32))
        acc = (acc_real + acc_fake)/2

        grads = tape.gradient(d_loss, self.discriminator.trainable_weights)
        d_grads = [g * d_train_condition for g in grads]
        self.d_optimizer.apply_gradients(zip(d_grads, self.discriminator.trainable_weights))

        # Treinar Gerador
        misleading_labels = tf.ones((batch_size, 1))

        with tf.GradientTape() as tape:
            # Gerar sequências dentro do tape
            generated_seqs = self.generator([random_latent_vectors, conditions])
            pred_fake = self.discriminator([generated_seqs, conditions])
            g_gan_loss = self.loss_fn(misleading_labels, pred_fake)
            g_loss = g_gan_loss
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

            #distancia do ponto mais próximo
            closest_x = tf.clip_by_value(px, x_min, x_max)
            closest_y = tf.clip_by_value(py, y_min, y_max)
            dist_x = tf.abs(px - closest_x)
            dist_y = tf.abs(py - closest_y)

            #distancia do ponto central
            # dist_x = tf.abs(px - target_x)
            # dist_y = tf.abs(py - target_y)

            #quadrado da distancia
            dist_x = (dist_x + 1)**3
            dist_y = (dist_y + 1)**3
            dist_total = (dist_x + dist_y) - 2

            is_outside = tf.cast(tf.greater(dist_total, 0), tf.float32)
            aabb_distance_loss = tf.reduce_mean(dist_total*AABB_DISTANCE_LOSS_MULT + (is_outside * AABB_FIXED_LOSS))
            g_loss += aabb_distance_loss
            #calcular taxa de acerto do gerador em terminar dentro do botão
            within_x = tf.logical_and(px >= x_min, px <= x_max)
            within_y = tf.logical_and(py >= y_min, py <= y_max)
            is_hit = tf.logical_and(within_x, within_y)
            g_hit_rate = tf.reduce_mean(tf.cast(is_hit, tf.float32))
            g_hit_loss = 1 - g_hit_rate
            g_loss += g_hit_loss

        grads = tape.gradient(g_loss, self.generator.trainable_weights)
        self.g_optimizer.apply_gradients(zip(grads, self.generator.trainable_weights))

        self.d_loss_metric.update_state(d_loss)
        self.g_loss_metric.update_state(g_loss)
        
        return {
            "d_training": d_train_condition,
            "d_loss": self.d_loss_metric.result(),
            "g_loss": self.g_loss_metric.result(),
            "aabb_distance_loss": aabb_distance_loss,
            "g_gan_loss":g_gan_loss,
            "acc": acc,
            "acc_real": acc_real,
            "acc_fake": acc_fake,
            "g_hit": g_hit_rate
        }

if __name__ == "__main__":
    print(tf.config.list_physical_devices('GPU'))
    print("Cuda Disponível:", tf.test.is_built_with_cuda())

    X_seq, X_cond = load_padded_sequences()
    generator = None
    discriminator = None

    generator_load_path = ""
    discriminator_load_path = ""
    loaded_epoch = 0

    if LOAD_BEST:
        folder_path = get_best_model()
        loaded_epoch = int(folder_path.split('\\')[-1])
        generator_load_path = f"{folder_path}mouse_gan_generator.keras"
        discriminator_load_path = f"{folder_path}mouse_gan_discriminator.keras"
    elif EPOCH_TO_LOAD == 0:
        generator_load_path = f"{get_base_path()}models/mouse_gan_generator.keras"
        discriminator_load_path = f"{get_base_path()}models/mouse_gan_discriminator.keras"
    else:
        loaded_epoch = EPOCH_TO_LOAD
        generator_load_path = f"{get_base_path()}models/{EPOCH_TO_LOAD}/mouse_gan_generator.keras"
        discriminator_load_path = f"{get_base_path()}models/{EPOCH_TO_LOAD}/mouse_gan_discriminator.keras"

    print("-----------------")

    try:
        generator = models.load_model(generator_load_path)
    except Exception as e:
        print(e)
        print("CREATING NEW GENERATOR")
        generator = build_generator()
    try:
        if not LOAD_DISCRIMINATOR:
            raise Exception("dont load discriminator")
        discriminator = models.load_model(discriminator_load_path)
    except Exception as e:
        print(e)
        print("CREATING NEW DISCRIMINATOR")
        discriminator = build_discriminator()

    print("-----------------")

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
        callbacks=[GANMonitor(LATENT_DIM, loaded_epoch)]
    )

    