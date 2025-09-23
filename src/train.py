import argparse, os, torch
from torch import nn, optim
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from model import DualBranchSwinCNNClassifier
from dataset import get_dataloaders, get_dataloaders_auto

def eval_epoch(model, loader, device):
    model.eval(); ys, ps = [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            ps.extend(model(x).argmax(1).cpu().tolist())
            ys.extend(y.cpu().tolist())
    return accuracy_score(ys, ps), ys, ps

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.auto_split:
        tr, va, classes = get_dataloaders_auto(args.data, args.batch_size, args.val_ratio, args.seed)
    else:
        tr, va, classes = get_dataloaders(os.path.join(args.data,"train"), os.path.join(args.data,"val"), args.batch_size)

    model = DualBranchSwinCNNClassifier(num_classes=args.num_classes, pretrained=True).to(device)
    crit = nn.CrossEntropyLoss(); opt = optim.Adam(model.parameters(), lr=args.lr)

    for e in range(1, args.epochs+1):
        model.train(); run = 0.0
        for x,y in tr:
            x,y = x.to(device), y.to(device)
            opt.zero_grad(); out = model(x); loss = crit(out,y)
            loss.backward(); opt.step(); run += loss.item()*x.size(0)
        tr_loss = run/len(tr.dataset)
        val_acc,_,_ = eval_epoch(model, va, device)
        print(f"Epoch {e:02d}/{args.epochs} | Train Loss {tr_loss:.4f} | Val Acc {val_acc:.4f}")

    val_acc, ys, ps = eval_epoch(model, va, device)
    P,R,F1,_ = precision_recall_fscore_support(ys, ps, average="macro", zero_division=0)
    print(f"[final] Val Acc {val_acc:.4f} | Macro P/R/F1 {P:.4f}/{R:.4f}/{F1:.4f}")

    torch.save(model.state_dict(), args.output_model)
    print(f"[save] {args.output_model}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="./data")
    ap.add_argument("--auto_split", action="store_true")
    ap.add_argument("--val_ratio", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num_classes", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--output_model", default="student_model.pth")
    args = ap.parse_args()
    train(args)

if __name__ == "__main__":
    main()
