from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import resample_poly, spectrogram

from sweep import ЧАСТОТА, анализ, импульсная, метрики, свип, эталон_и_полоса

СЕРИИ = {
    "Tube Screamer (мягкое)": {3: "гитара_ts_g3", 33: "гитара_ts_g33", 66: "гитара_ts_g66", 100: "гитара_ts_g100"},
    "RAT (жёсткое)": {6: "гитара_rat_d6", 33: "гитара_rat_d33", 66: "гитара_rat_d66", 100: "гитара_rat_d100"},
}


def посчитать(метка):
    _, запись = wavfile.read(f"data/{метка}.wav")
    # ponytail: внешний режим узнаём по длине записи (60 с против 15); появится третий режим — хранить флаг рядом с wav
    эталон, верх = эталон_и_полоса(запись.size > 30 * ЧАСТОТА)
    *_, граница, их, _ = анализ(запись, эталон, ЧАСТОТА, верх)
    return граница, np.abs(запись).max(), *метрики(их, ЧАСТОТА, верх, запись.size - эталон.size)


def ячейка(уровень, шум):
    return f"{уровень:.1f}" if уровень > шум + 6 else "шум"


def таблица():
    print("| канал | граница, кГц | пик | шум гармоник | 2-я | 3-я | 5-я | 7-я | RT60, с | после 64 мс, дБ |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for файл in sorted(Path("channels").glob("*.npy")):
        граница, пик, г, шум, rt60, поздняя = посчитать(файл.stem)
        гармоники = " | ".join(ячейка(г[i], шум) for i in (0, 1, 3, 5))
        rt = "—" if "delay" in файл.stem else f"{rt60:.2f}"
        поздно = "шум" if поздняя < -100 else f"{поздняя:.1f}"
        print(f"| {файл.stem} | {граница / 1000:.1f} | {пик:.2f} | {шум:.1f} | {гармоники} | {rt} | {поздно} |")


def график_серий():
    _, оси = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    for ось, (название, серия) in zip(оси, СЕРИИ.items()):
        замеры = [посчитать(метка) for метка in серия.values()]
        уровни = np.array([з[2] for з in замеры])
        шум = max(з[3] for з in замеры)
        ось.axhline(шум, color="gray", ls=":", label="шум")
        for номер, маркер in ((1, "o-"), (3, "s-"), (5, "^-")):
            ось.plot(list(серия), уровни[:, номер], маркер, label=f"{номер + 2}-я гармоника")
        ось.set(title=название, xlabel="Gain / Distortion", ylim=(-80, 0))
        ось.grid(alpha=0.3)
        ось.legend()
    оси[0].set_ylabel("дБ относительно основного тона")
    plt.tight_layout()
    plt.savefig("figures/гармоники_ts_rat.png", dpi=150)


def график_ревербов():
    _, оси = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
    for ось, метка in zip(оси, ("гитара_room", "гитара_hall", "гитара_spring")):
        _, запись = wavfile.read(f"data/{метка}.wav")
        их = импульсная(запись, свип(ЧАСТОТА))
        пик = np.argmax(np.abs(их))
        отрезок = resample_poly(их[пик - ЧАСТОТА // 100 : пик + int(0.6 * ЧАСТОТА)], 1, 4)
        ч, т, мощность = spectrogram(отрезок, ЧАСТОТА // 4, nperseg=512, noverlap=480)
        дб = 10 * np.log10(мощность + 1e-20)
        ось.pcolormesh(т * 1000 - 10, ч / 1000, дб, shading="auto", vmin=дб.max() - 60)
        ось.set(title=метка, xlabel="мс после прямого звука", ylim=(0, 6))
    оси[0].set_ylabel("кГц")
    plt.tight_layout()
    plt.savefig("figures/ревербы_спектрограммы.png", dpi=110)


if __name__ == "__main__":
    таблица()
    график_серий()
    график_ревербов()
