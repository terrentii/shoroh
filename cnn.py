import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from torch import nn

from simulator import ОТВОДОВ, СЦЕНАРИИ, ber_приёмника, канал_из_их, mmse, принять, qpsk_демод, qpsk_мод, q_теория

устройство = "cuda" if torch.cuda.is_available() else "cpu"
ЗАДЕРЖКА = ОТВОДОВ // 4
ПОТОКОВ, СИМВОЛОВ = 32, 2**16
ШАГОВ, ПАКЕТ = 20000, 2048
ОБУЧЕНИЕ_MMSE = 2**16
ШКАЛА_ДБ = np.arange(0, 31, 3)


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


def обучить(канал, искажение):
    сигналы, цели, начала = [], [], []
    for номер in range(ПОТОКОВ):
        дополнено, биты = поток(канал, искажение, np.random.uniform(0, 30), СИМВОЛОВ)
        сдвиг = sum(с.shape[1] for с in сигналы)
        сигналы.append(на_gpu(дополнено))
        цели.append(torch.tensor(биты, dtype=torch.float32, device=устройство))
        начала.append(сдвиг + torch.arange(СИМВОЛОВ, device=устройство))
    сигнал, цели, начала = torch.cat(сигналы, 1), torch.cat(цели), torch.cat(начала)
    модель = Приёмник().to(устройство)
    оптимизатор = torch.optim.Adam(модель.parameters(), 1e-3)
    расписание = torch.optim.lr_scheduler.CosineAnnealingLR(оптимизатор, ШАГОВ)
    потери = nn.BCEWithLogitsLoss()
    for шаг in range(ШАГОВ):
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


def сравнить(название):
    метка, искажение = СЦЕНАРИИ[название]
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


if __name__ == "__main__":
    torch.manual_seed(0)
    np.random.seed(0)
    for название in sys.argv[1:] or СЦЕНАРИИ:
        сравнить(название)
