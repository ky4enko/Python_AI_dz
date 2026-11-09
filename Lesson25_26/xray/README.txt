PneumoniaMNIST 64×64 — рентгени грудної клітки, клас 0 = NORMAL, 1 = PNEUMONIA.
Джерело: MedMNIST v2 (Yang et al., Scientific Data 2023), ліцензія CC BY 4.0,
похідний від Kermany et al. 2018 / Kaggle «Chest X-Ray Images (Pneumonia)».

pneumoniamnist_64.npz: train_images (4708,64,64), val_images (524), test_images (624)
                       + *_labels (N,1) uint8
png/ — 10 NORMAL + 10 PNEUMONIA з test у рідній роздільності 64×64.
Split по знімках, як у MedMNIST; у клініці треба ділити по пацієнтах.
