from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import fftconvolve, resample_poly
from numpy.lib.stride_tricks import sliding_window_view
from scipy.special import erfc

гсч = np.random.default_rng(0)
ПОДНЕСУЩИХ, ПРЕФИКС = 1024, 256
ПОЛОСА = 4000
ПЕРЕДИСКРЕТИЗАЦИЯ = 16
ПИЛОТОВ = 4
ОБУЧЕНИЕ, ОТВОДОВ = 4096, 129
# подобрано по 3-й, 5-й и 7-й гармоникам из Э10 (report.подобрать_искажения): RAT 66/100 с точностью 0.3 дБ, TS 100 — 2.7 дБ
ИСКАЖЕНИЯ = {
    "TS33": (1.192, 4.83),
    "TS66": (1.702, 1.08),
    "TS100": (0.739, 1.53),
    "RAT33": (1.294, 2.45),
    "RAT66": (0.675, 1.78),
    "RAT100": (0.491, 3.61),
}


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


def рапп(x, насыщение, плавность):
    return x / (1 + np.abs(x / насыщение) ** (2 * плавность)) ** (1 / (2 * плавность))


def вверх(x, во_сколько):
    спектр = np.fft.fft(x)
    широкий = np.zeros(x.size * во_сколько, complex)
    широкий[: x.size // 2] = спектр[: x.size // 2]
    широкий[x.size // 2 - x.size :] = спектр[x.size // 2 :]
    return np.fft.ifft(широкий) * во_сколько


def вниз(y, во_сколько):
    спектр, n = np.fft.fft(y), y.size // во_сколько
    return np.fft.ifft(np.concatenate([спектр[: n // 2], спектр[n // 2 - n :]])) / во_сколько


def исказить(сигнал, насыщение, плавность, несущая):
    время = np.arange(сигнал.size * ПЕРЕДИСКРЕТИЗАЦИЯ) / (ПОЛОСА * ПЕРЕДИСКРЕТИЗАЦИЯ)
    сдвиг = np.exp(2j * np.pi * несущая * время)
    звук = np.real(вверх(сигнал, ПЕРЕДИСКРЕТИЗАЦИЯ) * сдвиг)
    звук /= np.sqrt(2 * np.mean(звук**2))
    обратно = вниз(рапп(звук, насыщение, плавность) * np.conj(сдвиг), ПЕРЕДИСКРЕТИЗАЦИЯ)
    return обратно / np.sqrt(np.mean(np.abs(обратно) ** 2))


def принять(сигнал, EbN0_дб, бит_на_символ, канал, искажение, несущая):
    if искажение:
        сигнал = исказить(сигнал, *ИСКАЖЕНИЯ[искажение], несущая)
    if канал.size > 1:
        сигнал = fftconvolve(сигнал, канал)[: сигнал.size]
    return awgn(сигнал, EbN0_дб, бит_на_символ)


def mmse(принято, обучение):
    задержка = ОТВОДОВ // 4
    дополнено = np.concatenate([np.zeros(ОТВОДОВ - 1 - задержка), принято, np.zeros(задержка)])
    окна = sliding_window_view(дополнено[: обучение.size + ОТВОДОВ - 1], ОТВОДОВ)
    фильтр = np.linalg.lstsq(окна, обучение, rcond=None)[0]
    return fftconvolve(дополнено, фильтр[::-1], "valid")


def оценка_канала(принято, пилоты):
    время = принято.reshape(-1, ПОДНЕСУЩИХ + ПРЕФИКС)[:ПИЛОТОВ, ПРЕФИКС:]
    отклик = np.fft.ifft(np.mean(np.fft.fft(время, norm="ortho") / пилоты, axis=0))
    отклик[ПРЕФИКС:-16] = 0
    return np.fft.fft(отклик)


# ponytail: энергия пилотов и обучающей последовательности не входит в Eb/N0 — это ~0.8% и 1.5% от передачи
def ber_приёмника(приёмник, EbN0_дб, канал=np.ones(1), искажение=None, несущая=2000, число_бит=2**19):
    данные = гсч.integers(0, 2, число_бит)
    if приёмник.startswith("OFDM"):
        пилот_биты = гсч.integers(0, 2, 2 * ПИЛОТОВ * ПОДНЕСУЩИХ)
        принято = принять(ofdm_мод(np.concatenate([пилот_биты, данные])), EbN0_дб, 2, канал, искажение, несущая)
        if приёмник == "OFDM, пилоты":
            передаточная = оценка_канала(принято, qpsk_мод(пилот_биты).reshape(ПИЛОТОВ, -1))
        else:
            передаточная = np.fft.fft(канал, ПОДНЕСУЩИХ)
        return np.mean(ofdm_демод(принято, передаточная)[пилот_биты.size :] != данные)
    обучение_биты = гсч.integers(0, 2, 2 * ОБУЧЕНИЕ)
    символы = qpsk_мод(np.concatenate([обучение_биты, данные]))
    принято = принять(символы, EbN0_дб, 2, канал, искажение, несущая)
    принято = mmse(принято, символы[:ОБУЧЕНИЕ]) if приёмник == "QPSK, MMSE" else принято / канал[0]
    return np.mean(qpsk_демод(принято)[обучение_биты.size :] != данные)


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
    # ponytail: у звукоснимателя полоса до ~4 кГц, поэтому гитарные каналы только на 2 кГц; появятся другие узкополосные — хранить полосу рядом с каналом
    for несущая in (2000,) if метка.startswith("гитара") else (5000, 20000):
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
    plt.close()


СЦЕНАРИИ = {
    "Spring (линейный)": ("гитара_spring", None),
    "RAT 100": ("гитара_rat_d100", "RAT100"),
    "RAT 100 + Spring": ("гитара_spring", "RAT100"),
    "TS 100 + Spring": ("гитара_spring", "TS100"),
}
ПРИЁМНИКИ = {"QPSK, без эквалайзера": "o:", "QPSK, MMSE": "o-", "OFDM, пилоты": "s-", "OFDM, идеальный канал": "s--"}


def график_честной_классики(частота_дискр=192000):
    шкала_дб = np.arange(0, 31, 3)
    _, оси = plt.subplots(2, 2, figsize=(12, 9), sharex=True, sharey=True)
    for ось, (название, (метка, искажение)) in zip(оси.ravel(), СЦЕНАРИИ.items()):
        канал = канал_из_их(np.load(f"channels/{метка}.npy"), частота_дискр, 2000)
        ось.semilogy(шкала_дб, q_теория(10 ** (шкала_дб / 10)), "k--", lw=1, label="AWGN, теория")
        for приёмник, стиль in ПРИЁМНИКИ.items():
            точки = [max(ber_приёмника(приёмник, дб, канал, искажение), 1e-6) for дб in шкала_дб]
            ось.semilogy(шкала_дб, точки, стиль, label=приёмник)
            print(название, приёмник, [f"{т:.1g}" for т in точки], flush=True)
        ось.set(title=название, ylim=(1e-6, 1))
        ось.grid(which="both", alpha=0.3)
    for ось in оси[1]:
        ось.set_xlabel("Eb/N0, дБ")
    for ось in оси[:, 0]:
        ось.set_ylabel("BER")
    оси[0, 0].legend(fontsize=8)
    plt.tight_layout()
    plt.savefig("figures/ber_классика.png", dpi=150)


if __name__ == "__main__":
    Path("figures").mkdir(exist_ok=True)
    график_awgn()
    for файл in Path("channels").glob("*.npy"):
        график_комнаты(файл.stem)
    график_честной_классики()
