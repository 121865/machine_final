import numpy as np
import matplotlib.pyplot as plt


def load_loss_csv(path: str = "loss_log_sf_v3.csv"):
    """
    讀取訓練時記錄的 loss CSV，格式可能是：
      新版：step,actor_loss,critic1_loss,critic2_loss,alpha_loss,alpha_value
      舊版：step,actor_loss,critic1_loss,critic2_loss,alpha_loss
    """
    steps = []
    actor_losses = []
    critic1_losses = []
    critic2_losses = []
    alpha_losses = []
    alpha_values = []

    with open(path, "r", encoding="utf-8") as f:
        header = f.readline()
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")

            # 至少要有 step + 4 個 loss
            if len(parts) < 5:
                continue

            try:
                step = int(parts[0])
                a = float(parts[1])
                c1 = float(parts[2])
                c2 = float(parts[3])
                al = float(parts[4])
                if len(parts) >= 6:
                    av = float(parts[5])
                else:
                    av = np.nan
            except ValueError:
                continue

            steps.append(step)
            actor_losses.append(a)
            critic1_losses.append(c1)
            critic2_losses.append(c2)
            alpha_losses.append(al)
            alpha_values.append(av)

    return (
        np.array(steps),
        np.array(actor_losses),
        np.array(critic1_losses),
        np.array(critic2_losses),
        np.array(alpha_losses),
        np.array(alpha_values),
    )


def main():
    steps, actor_losses, critic1_losses, critic2_losses, alpha_losses, alpha_values = load_loss_csv()

    print("loaded points:", len(steps))
    if len(steps) == 0:
        print("沒有讀到任何資料，請檢查 CSV 路徑與格式。")
        return

    # ---- Critic ----
    plt.figure()
    plt.plot(steps, critic1_losses, label="critic1_loss")
    plt.plot(steps, critic2_losses, label="critic2_loss")
    plt.xlabel("Training step")
    plt.ylabel("Loss")
    plt.title("Critic losses over training")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("critic_loss_curve_sf_v3.png")
    print("已將 critic loss 曲線存成 critic_loss_curve_sf_v3.png")

    # ---- Actor ----
    plt.figure()
    plt.plot(steps, actor_losses, label="actor_loss")
    plt.xlabel("Training step")
    plt.ylabel("Loss")
    plt.title("Actor loss over training")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("actor_loss_curve_sf_v3.png")
    print("已將 actor loss 曲線存成 actor_loss_curve_sf_v3.png")

    # ---- Alpha loss ----
    plt.figure()
    plt.plot(steps, alpha_losses, label="alpha_loss")
    plt.xlabel("Training step")
    plt.ylabel("Loss")
    plt.title("Alpha loss over training")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("alpha_loss_curve_sf_v3.png")
    print("已將 alpha loss 曲線存成 alpha_loss_curve_sf_v3.png")

    # ---- Alpha value（真正的 α）----
    if not np.all(np.isnan(alpha_values)):
        plt.figure()
        plt.plot(steps, alpha_values, label="alpha_value")
        plt.xlabel("Training step")
        plt.ylabel("alpha")
        plt.title("Alpha value over training")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("alpha_value_curve_sf_v3.png")
        print("已將 alpha value 曲線存成 alpha_value_curve_sf_v3.png")

    plt.show()


if __name__ == "__main__":
    main()
