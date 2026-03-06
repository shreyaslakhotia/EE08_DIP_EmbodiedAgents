# How to Access GPU Cluster

## 1. Open a Terminal

- **Windows**: Use PowerShell, Command Prompt, or WSL
- **macOS / Linux**: Open Terminal

## 2. SSH into the Cluster

```bash
ssh s125mdg25_08@gpu25.dynip.ntu.edu.sg
```

- **First time only**: Type `yes` and press Enter
- Enter your password when prompted

## 3. You're Logged In

You'll see a prompt like:

```
s125mdg25_08@gpu25:~$
```

## 4. Basic Commands

```bash
pwd        # show current directory
ll         # list files
cd folder  # change directory
```

## 5. Check GPU Status

```bash
nvidia-smi
```

**Note**: Use only 1–2 GPUs, as per policy.

## 6. Run Your Work

Example:

```bash
python train.py
```

Clear GPU memory when done:

```bash
exit
```

## 7. Transfer Files

### From Local → Cluster

```bash
scp file.py s125mdg25_08@gpu25.dynip.ntu.edu.sg:~
```

### From Cluster → Local

```bash
scp s125mdg25_08@gpu25.dynip.ntu.edu.sg:~/file.py .
```

## 8. Exit the Cluster

```bash
exit
```
