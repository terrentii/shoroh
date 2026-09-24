import subprocess
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import chirp, resample_poly, spectrogram
from scipy.signal.windows import tukey

ЧАСТОТА = 192000
ТИШИНА, ДЛИТЕЛЬНОСТЬ = 2, 8
ГРОМКОСТЬ = 0.5
ЗАПАС_ДБ = 10
ОКНО_ИХ = 0.05
ХВОСТ_КАНАЛА = 0.25
УСТРОЙСТВО = "pipewire"


def свип(частота_дискр):
    время = np.arange(ДЛИТЕЛЬНОСТЬ * частота_дискр) / частота_дискр
    сигнал = chirp(время, 20, ДЛИТЕЛЬНОСТЬ, 0.45 * частота_дискр, method="logarithmic")
    отсчёты = np.arange(сигнал.size)
    огибающая = np.minimum(1, np.minimum(отсчёты, отсчёты[::-1]) / (0.01 * частота_дискр))
    return (ГРОМКОСТЬ * сигнал * огибающая).astype(np.float32)


def частота_pipewire(значение):
    subprocess.run(["pw-metadata", "-n", "settings", "0", "clock.force-rate", str(значение)], capture_output=True)


def записать(внешний):
    import sounddevice as sd

    sd.default.device = УСТРОЙСТВО
    частота_pipewire(ЧАСТОТА)
    try:
        if внешний:
            wavfile.write("sweep_48k.wav", 48000, свип(48000))
            input("Скопируй sweep_48k.wav на устройство и нажми Enter")
            запись = sd.rec((ТИШИНА + ДЛИТЕЛЬНОСТЬ + 5) * ЧАСТОТА, ЧАСТОТА, channels=2)
            sd.sleep(ТИШИНА * 1000)
            print("Включай!")
            sd.wait()
        else:
            тишина = np.zeros(ТИШИНА * ЧАСТОТА, np.float32)
            сигнал = np.concatenate([тишина, свип(ЧАСТОТА), тишина])
            запись = sd.playrec(np.column_stack([сигнал, сигнал]), ЧАСТОТА, channels=2, blocking=True)
    finally:
        частота_pipewire(0)
    return запись[:, 0]


def импульсная(запись, эталон):
    n = запись.size + эталон.size
    спектр_записи, спектр_эталона = np.fft.rfft(запись, n), np.fft.rfft(эталон, n)
    мощность = np.abs(спектр_эталона) ** 2
    return np.fft.irfft(спектр_записи * np.conj(спектр_эталона) / (мощность + 1e-6 * мощность.max()), n)


# ponytail: сглаживание скользящим средним по 32 бинам (~600 Гц), низы размываются; нужна точная АЧХ ниже 1 кГц — дробнооктавное сглаживание
def сгладить_дб(окно):
    мощность = np.abs(np.fft.rfft(окно * tukey(окно.size, 0.2))) ** 2
    return 10 * np.log10(np.convolve(мощность, np.ones(32) / 32, "same") + 1e-30)


def анализ(запись, эталон, частота_дискр, верх_полосы):
    их = импульсная(запись, эталон)
    пик = np.argmax(np.abs(их))
    до, после = int(0.005 * частота_дискр), int(ОКНО_ИХ * частота_дискр)
    сдвиг = пик + частота_дискр // 2
    ачх = сгладить_дб(их[пик - до : пик + после])
    шум = сгладить_дб(их[сдвиг - до : сдвиг + после])
    частоты = np.fft.rfftfreq(до + после, 1 / частота_дискр)
    годные = (частоты <= верх_полосы) & (ачх - шум > ЗАПАС_ДБ)
    return частоты, ачх, шум, частоты[годные].max(initial=0), их, пик


def график(запись, эталон, частота_дискр, верх_полосы, путь):
    частоты, ачх, шум, граница, их, пик = анализ(запись, эталон, частота_дискр, верх_полосы)
    _, (верх, середина, низ) = plt.subplots(3, 1, figsize=(10, 11))
    верх.semilogx(частоты[1:], ачх[1:], label="АЧХ канала")
    верх.semilogx(частоты[1:], шум[1:], label="шум")
    верх.axvline(граница, color="k", ls="--", label=f"граница {граница / 1000:.1f} кГц")
    верх.set(xlabel="Частота, Гц", ylabel="дБ")
    верх.legend()
    верх.grid(which="both", alpha=0.3)
    отрезок = их[пик - int(1.6 * частота_дискр) : пик + int(0.2 * частота_дискр)]
    время = (np.arange(отрезок.size) - int(1.6 * частота_дискр)) / частота_дискр * 1000
    середина.plot(время, 20 * np.log10(np.abs(отрезок) / np.abs(их[пик]) + 1e-12), lw=0.5)
    середина.set(xlabel="Время относительно прямого звука, мс", ylabel="Импульсная характеристика, дБ", ylim=(-100, 5))
    середина.grid(alpha=0.3)
    ч, т, мощность = spectrogram(запись, частота_дискр, nperseg=4096)
    низ.pcolormesh(т, ч / 1000, 10 * np.log10(мощность + 1e-20), shading="auto")
    низ.set(xlabel="Время, с", ylabel="Частота, кГц")
    plt.tight_layout()
    plt.savefig(путь, dpi=120)
    return граница, их, пик


if __name__ == "__main__":
    метка = sys.argv[1]
    внешний = "--внешний" in sys.argv
    данные = Path("data")
    данные.mkdir(exist_ok=True)
    if "--из-файла" in sys.argv:
        _, запись = wavfile.read(данные / f"{метка}.wav")
    else:
        запись = записать(внешний)
        wavfile.write(данные / f"{метка}.wav", ЧАСТОТА, запись)
    # ponytail: у внешнего устройства свои часы, дрейф частоты размоет ИХ на длинном свипе; заметно — оценивать дрейф по пику корреляции
    эталон = resample_poly(свип(48000), ЧАСТОТА // 48000, 1) if внешний else свип(ЧАСТОТА)
    верх_полосы = 0.45 * (48000 if внешний else ЧАСТОТА)
    граница, их, пик = график(запись, эталон, ЧАСТОТА, верх_полосы, данные / f"{метка}.png")
    Path("channels").mkdir(exist_ok=True)
    np.save(Path("channels") / f"{метка}.npy", их[пик - int(0.005 * ЧАСТОТА) : пик + int(ХВОСТ_КАНАЛА * ЧАСТОТА)].astype(np.float32))
    print(f"{метка}: граница {граница / 1000:.1f} кГц, пик записи {np.abs(запись).max():.2f}")
