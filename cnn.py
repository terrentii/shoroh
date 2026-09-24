import json
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from torch import nn

from simulator import ИСКАЖЕНИЯ, ОТВОДОВ, СЦЕНАРИИ, исказить, ber_приёмника, канал_из_их, mmse, принять, qpsk_демод, qpsk_мод, q_теория

устройство = "cuda" if torch.cuda.is_available() else "cpu"
ЗАДЕРЖКА = ОТВОДОВ // 4
ПОТОКОВ, СИМВОЛОВ = 32, 2**16
ШАГОВ, ПАКЕТ = 60000, 2048
ДИАПАЗОН_ДБ = (0, 35)
ОБНОВЛЯТЬ_КАЖДЫЕ = 2000
ОБУЧЕНИЕ_MMSE = 2**16
ШКАЛА_ДБ = np.arange(0, 31, 3)
СЕРИЯ = {
    "без искажений": ("гитара_чисто", None, None),
    "TS 33": ("гитара_ts_g33", "TS33", -44.9),
    "TS 66": ("гитара_ts_g66", "TS66", -29.8),
    "TS 100": ("гитара_ts_g100", "TS100", -14.5),
    "RAT 33": ("гитара_rat_d33", "RAT33", -35.7),
    "RAT 66": ("гитара_rat_d66", "RAT66", -17.2),
    "RAT 100": ("гитара_rat_d100", "RAT100", -13.7),
}


class Приёмник(nn.Module):
    def __init__(self, каналов=64):
        super().__init__()
        слои = [nn.Conv1d(2, каналов, 7, padding=3), nn.GELU()]
        for расширение in (2, 4, 8):
            слои += [nn.Conv1d(каналов, каналов, 7, padding=3 * расширение, dilation=расширение), nn.GELU()]
        self.свёртки = nn.Sequential(*слои)
        self.нелинейный = nn.Linear(каналов * ОТВОДОВ, 2)
        self.линейный = nn.Linear(2 * ОТВОДОВ, 2)

    def forward(self, окна):
        return self.нелинейный(self.свёртки(окна).flatten(1)) + self.линейный(окна.flatten(1))


def поток(канал, искажение, EbN0_дб, символов):
    биты = np.random.randint(0, 2, 2 * символов)
    принято = принять(qpsk_мод(биты), EbN0_дб, 2, канал, искажение, 2000)
    дополнено = np.concatenate([np.zeros(ОТВОДОВ - 1 - ЗАДЕРЖКА), принято, np.zeros(ЗАДЕРЖКА)])
    return дополнено, биты.reshape(-1, 2)


def на_gpu(дополнено):
    return torch.tensor(np.stack([дополнено.real, дополнено.imag]), dtype=torch.float32, device=устройство)


def окна(сигнал, начала):
    return сигнал[:, начала[:, None] + torch.arange(ОТВОДОВ, device=устройство)].permute(1, 0, 2)


def датасет(канал, искажение, диапазон_дб):
    сигналы, цели, начала = [], [], []
    for номер in range(ПОТОКОВ):
        дополнено, биты = поток(канал, искажение, np.random.uniform(*диапазон_дб), СИМВОЛОВ)
        сдвиг = sum(с.shape[1] for с in сигналы)
        сигналы.append(на_gpu(дополнено))
        цели.append(torch.tensor(биты, dtype=torch.float32, device=устройство))
        начала.append(сдвиг + torch.arange(СИМВОЛОВ, device=устройство))
    return torch.cat(сигналы, 1), torch.cat(цели), torch.cat(начала)


def обучить(канал, искажение, диапазон_дб=ДИАПАЗОН_ДБ, шагов=ШАГОВ, обновлять_каждые=ОБНОВЛЯТЬ_КАЖДЫЕ):
    модель = Приёмник().to(устройство)
    оптимизатор = torch.optim.Adam(модель.parameters(), 1e-3)
    расписание = torch.optim.lr_scheduler.CosineAnnealingLR(оптимизатор, шагов)
    потери = nn.BCEWithLogitsLoss()
    for шаг in range(шагов):
        if шаг == 0 or обновлять_каждые and шаг % обновлять_каждые == 0:
            сигнал, цели, начала = датасет(канал, искажение, диапазон_дб)
        выбор = torch.randint(0, начала.numel(), (ПАКЕТ,), device=устройство)
        ошибка = потери(модель(окна(сигнал, начала[выбор])), цели[выбор])
        оптимизатор.zero_grad()
        ошибка.backward()
        оптимизатор.step()
        расписание.step()
        if шаг % 2000 == 0:
            print(f"  шаг {шаг}: потери {ошибка.item():.4f}", flush=True)
    return модель


@torch.no_grad()
def ber_cnn(модель, канал, искажение, EbN0_дб, символов=2**18):
    дополнено, биты = поток(канал, искажение, EbN0_дб, символов)
    сигнал = на_gpu(дополнено)
    решения = torch.cat([модель(окна(сигнал, часть)) > 0 for часть in torch.arange(символов, device=устройство).split(8192)])
    return (решения.cpu().numpy() != биты).mean()


def ber_mmse_сытый(канал, искажение, EbN0_дб, символов=2**18):
    биты = np.random.randint(0, 2, 2 * (ОБУЧЕНИЕ_MMSE + символов))
    символы_ = qpsk_мод(биты)
    выровнено = mmse(принять(символы_, EbN0_дб, 2, канал, искажение, 2000), символы_[:ОБУЧЕНИЕ_MMSE])
    return np.mean(qpsk_демод(выровнено)[2 * ОБУЧЕНИЕ_MMSE :] != биты[2 * ОБУЧЕНИЕ_MMSE :])


def сравнить(название, метка, искажение):
    канал = канал_из_их(np.load(f"channels/{метка}.npy"), 192000, 2000)
    print(название, flush=True)
    модель = обучить(канал, искажение)
    Path("models").mkdir(exist_ok=True)
    torch.save(модель.state_dict(), f"models/{название}.pt")
    кривые = {
        "QPSK, CNN": [ber_cnn(модель, канал, искажение, дб) for дб in ШКАЛА_ДБ],
        "QPSK, MMSE (65536 символов обучения)": [ber_mmse_сытый(канал, искажение, дб) for дб in ШКАЛА_ДБ],
        "OFDM, пилоты": [ber_приёмника("OFDM, пилоты", дб, канал, искажение) for дб in ШКАЛА_ДБ],
    }
    plt.figure(figsize=(8, 5))
    plt.semilogy(ШКАЛА_ДБ, q_теория(10 ** (ШКАЛА_ДБ / 10)), "k--", lw=1, label="AWGN, теория")
    for (подпись, точки), стиль in zip(кривые.items(), ("o-", "o--", "s-")):
        plt.semilogy(ШКАЛА_ДБ, np.maximum(точки, 1e-6), стиль, label=подпись)
        print(f"  {подпись}: {[f'{т:.2g}' for т in точки]}", flush=True)
    plt.ylim(1e-6, 1)
    plt.xlabel("Eb/N0, дБ")
    plt.ylabel("BER")
    plt.title(название)
    plt.grid(which="both", alpha=0.3)
    plt.legend()
    plt.savefig(f"figures/cnn_{название}.png", dpi=150)
    plt.close()
    Path("results").mkdir(exist_ok=True)
    Path(f"results/{название}.json").write_text(json.dumps({к: [float(т) for т in в] for к, в in кривые.items()}, ensure_ascii=False))


def сигнал_искажение(искажение, сигнал):
    сигнал = сигнал / np.sqrt(np.mean(np.abs(сигнал) ** 2))
    выход = исказить(сигнал, *ИСКАЖЕНИЯ[искажение], 2000)
    усиление = np.vdot(сигнал, выход) / np.vdot(сигнал, сигнал)
    return 10 * np.log10(np.abs(усиление) ** 2 / np.mean(np.abs(выход - усиление * сигнал) ** 2))


def нужный_snr(точки, цель=1e-3):
    lg = np.log10(np.maximum(точки, 1e-9))
    ниже = np.nonzero(lg <= np.log10(цель))[0]
    if ниже.size == 0 or ниже[0] == 0:
        return np.nan
    i = ниже[0]
    return ШКАЛА_ДБ[i - 1] + (np.log10(цель) - lg[i - 1]) / (lg[i] - lg[i - 1]) * (ШКАЛА_ДБ[i] - ШКАЛА_ДБ[i - 1])


def график_серии():
    qpsk = qpsk_мод(np.random.randint(0, 2, 2**17))
    print("| сценарий | 3-я гармоника синуса, дБ | SDR QPSK, дБ | CNN | MMSE | OFDM, пилоты | выигрыш CNN над MMSE, дБ |")
    print("|---|---|---|---|---|---|---|")
    _, ось = plt.subplots(figsize=(8, 5))
    for семейство, маркер in (("TS", "o"), ("RAT", "s")):
        точки = []
        for название, (_, искажение, гармоника) in СЕРИЯ.items():
            if not название.startswith(семейство):
                continue
            кривые = json.loads(Path(f"results/{название}.json").read_text())
            нужно = [нужный_snr(кривые[к]) for к in кривые]
            sdr = сигнал_искажение(искажение, qpsk)
            точки.append((sdr, *нужно))
            print(f"| {название} | {гармоника} | {sdr:.1f} | {нужно[0]:.1f} | {нужно[1]:.1f} | {нужно[2]:.1f} | {нужно[1] - нужно[0]:.1f} |")
        точки = np.array(точки)
        ось.plot(точки[:, 0], точки[:, 2] - точки[:, 1], маркер + "-", label=f"{семейство}: выигрыш CNN над MMSE")
        ось.plot(точки[:, 0], точки[:, 3] - точки[:, 1], маркер + ":", label=f"{семейство}: выигрыш CNN над OFDM")
    чистый = json.loads(Path("results/без искажений.json").read_text())
    нужно = [нужный_snr(чистый[к]) for к in чистый]
    print(f"| без искажений | — | ∞ | {нужно[0]:.1f} | {нужно[1]:.1f} | {нужно[2]:.1f} | {нужно[1] - нужно[0]:.1f} |")
    ось.axhline(0, color="k", lw=0.8)
    ось.invert_xaxis()
    ось.set(xlabel="сигнал/искажения для QPSK, дБ (правее — сильнее нелинейность)", ylabel="выигрыш по Eb/N0 при BER 10⁻³, дБ", title="Чем сильнее нелинейность, тем больше выигрывает CNN")
    ось.grid(alpha=0.3)
    ось.legend()
    plt.tight_layout()
    plt.savefig("figures/cnn_серия.png", dpi=150)


if __name__ == "__main__":
    torch.manual_seed(0)
    np.random.seed(0)
    if sys.argv[1:] == ["серия"]:
        for название, (метка, искажение, _) in СЕРИЯ.items():
            if not Path(f"results/{название}.json").exists():
                сравнить(название, метка, искажение)
        график_серии()
    else:
        for название in sys.argv[1:] or СЦЕНАРИИ:
            сравнить(название, *СЦЕНАРИИ[название])
