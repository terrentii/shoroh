from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import fftconvolve, resample_poly
from scipy.special import erfc

гсч = np.random.default_rng(0)
ПОДНЕСУЩИХ, ПРЕФИКС = 1024, 256
ПОЛОСА = 4000


def awgn(сигнал, EbN0_дб, бит_на_символ):
    N0 = 1 / (бит_на_символ * 10 ** (EbN0_дб / 10))
    шум = гсч.standard_normal(сигнал.shape) + 1j * гсч.standard_normal(сигнал.shape)
    return сигнал + np.sqrt(N0 / 2) * шум


def bpsk_мод(биты):
    return (2 * биты - 1).astype(complex)


def bpsk_демод(сигнал):
    return (сигнал.real > 0).astype(int)


def qpsk_мод(биты):
    пары = 2 * биты.reshape(-1, 2) - 1
    return (пары[:, 0] + 1j * пары[:, 1]) / np.sqrt(2)


def qpsk_демод(сигнал):
    return np.column_stack([сигнал.real > 0, сигнал.imag > 0]).astype(int).ravel()


# ponytail: некогерентная FSK в пространстве сигналов (две ортогональные ветви со случайной фазой); волновая форма тонов понадобится для реального звука
def fsk_мод(биты):
    сигнал = np.zeros((биты.size, 2), complex)
    сигнал[np.arange(биты.size), биты] = np.exp(2j * np.pi * гсч.random(биты.size))
    return сигнал


def fsk_демод(сигнал):
    return (np.abs(сигнал[:, 1]) > np.abs(сигнал[:, 0])).astype(int)


def ofdm_мод(биты):
    символы = qpsk_мод(биты).reshape(-1, ПОДНЕСУЩИХ)
    время = np.fft.ifft(символы, norm="ortho")
    return np.hstack([время[:, -ПРЕФИКС:], время]).ravel()


def ofdm_демод(сигнал, передаточная=1):
    время = сигнал.reshape(-1, ПОДНЕСУЩИХ + ПРЕФИКС)[:, ПРЕФИКС:]
    return qpsk_демод((np.fft.fft(время, norm="ortho") / передаточная).ravel())


def канал_из_их(их, частота_дискр, несущая):
    время = np.arange(их.size) / частота_дискр
    узкая = resample_poly(их * np.exp(-2j * np.pi * несущая * время), 1, частота_дискр // ПОЛОСА)
    узкая = узкая[np.argmax(np.abs(узкая)) :]
    return узкая / np.linalg.norm(узкая)


def q_теория(EbN0):
    return 0.5 * erfc(np.sqrt(EbN0))


МОДУЛЯЦИИ = {
    "BPSK": (bpsk_мод, bpsk_демод, 1, q_теория),
    "QPSK": (qpsk_мод, qpsk_демод, 2, q_теория),
    "FSK": (fsk_мод, fsk_демод, 1, lambda EbN0: 0.5 * np.exp(-EbN0 / 2)),
    "OFDM": (ofdm_мод, ofdm_демод, 2, q_теория),
}


def ber(название, EbN0_дб, число_бит=2**20, канал=np.ones(1)):
    мод, демод, бит_на_символ, _ = МОДУЛЯЦИИ[название]
    биты = гсч.integers(0, 2, число_бит)
    сигнал = мод(биты)
    if канал.size > 1:
        сигнал = fftconvolve(сигнал, канал)[: сигнал.size]
    принято = awgn(сигнал, EbN0_дб, бит_на_символ)
    if название == "OFDM":
        оценка = ofdm_демод(принято, np.fft.fft(канал, ПОДНЕСУЩИХ))
    else:
        оценка = демод(принято / канал[0])
    return np.mean(оценка != биты)


def график_awgn():
    шкала_дб = np.arange(0, 10)
    plt.figure()
    for название, (*_, теория) in МОДУЛЯЦИИ.items():
        точки = [ber(название, дб, 2**21) for дб in шкала_дб]
        линия, = plt.semilogy(шкала_дб, точки, "o", label=f"{название} симуляция")
        plt.semilogy(шкала_дб, теория(10 ** (шкала_дб / 10)), "-", color=линия.get_color(), label=f"{название} теория")
    plt.xlabel("Eb/N0, дБ")
    plt.ylabel("BER")
    plt.grid(which="both", alpha=0.3)
    plt.legend()
    plt.savefig("figures/ber_awgn.png", dpi=150)


def график_комнаты(метка, частота_дискр=192000):
    их = np.load(f"channels/{метка}.npy")
    шкала_дб = np.arange(0, 32, 2)
    plt.figure(figsize=(8, 5))
    plt.semilogy(шкала_дб, q_теория(10 ** (шкала_дб / 10)), "k--", label="AWGN, теория")
    for несущая in (5000, 20000):
        канал = канал_из_их(их, частота_дискр, несущая)
        for название, маркер in (("QPSK", "o-"), ("OFDM", "s-")):
            точки = [max(ber(название, дб, канал=канал), 1e-7) for дб in шкала_дб]
            plt.semilogy(шкала_дб, точки, маркер, label=f"{название}, {несущая // 1000} кГц")
    plt.ylim(1e-6, 1)
    plt.xlabel("Eb/N0, дБ")
    plt.ylabel("BER")
    plt.title(f"Реальный канал «{метка}», полоса {ПОЛОСА // 1000} кГц")
    plt.grid(which="both", alpha=0.3)
    plt.legend()
    plt.savefig(f"figures/ber_{метка}.png", dpi=150)


if __name__ == "__main__":
    Path("figures").mkdir(exist_ok=True)
    график_awgn()
    for файл in Path("channels").glob("*.npy"):
        график_комнаты(файл.stem)
