import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import DataLoader
from torch_scatter import scatter_mean
from tqdm import tqdm

from gnn import *
from utils.argparser import argument_parser
from utils.util import load_data
import time



def size_of_model(model):
    torch.save(model.state_dict(), "temp.p")
    size=os.path.getsize("temp.p")/1e6
    print('Size (MB):', size)
    os.remove('temp.p')
    return size

def time_model_evaluation(model, device, train_loader, test1_loader, test2_loader, criterion):
    s = time.time()
    _ = test(
            model=model, device=device, train_data=train_loader, test_data1=test1_loader, test_data2=test2_loader, epoch=-1, criterion=criterion)
    elapsed = time.time() - s
    print("Elapsed time (seconds): {:.1f}".format(elapsed))
    return elapsed


def __loss_aux(output, loss, data, binary_prediction):
    if binary_prediction:
        labels = torch.zeros_like(output).scatter_(
            1, data.node_labels.unsqueeze(1), 1.)
    else:
        raise NotImplementedError()

    return loss(output, labels)


def train(
        model,
        device,
        training_data,
        optimizer,
        criterion,
        scheduler,
        binary_prediction=True) -> float:
    model.train()

    loss_accum = []

    for data in tqdm(training_data):
        data = data.to(device)

        output = model(x=data.x,
                       edge_index=data.edge_index,
                       batch=data.batch)

        loss = __loss_aux(
            output=output,
            loss=criterion,
            data=data,
            binary_prediction=binary_prediction)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        loss_accum.append(loss.detach().cpu().numpy())

    average_loss = np.mean(loss_accum)

    print(f"Train loss: {average_loss}")

    return average_loss, loss_accum


def __accuracy_aux(node_labels, predicted_labels, batch, device):

    results = torch.eq(
        predicted_labels,
        node_labels).type(
        torch.FloatTensor).to(device)

    # micro average -> mean between all nodes
    micro = torch.sum(results)

    # macro average -> mean between the mean of nodes for each graph
    macro = torch.sum(scatter_mean(results, batch))

    return micro, macro


def test(
        model,
        device,
        criterion,
        epoch,
        train_data,
        test_data1,
        test_data2=None,
        binary_prediction=True):
    model.eval()

    # ----- TRAIN ------
    train_micro_avg = 0.
    train_macro_avg = 0.

    if train_data is not None:
        n_nodes = 0
        n_graphs = 0
        for data in train_data:
            data = data.to(device)

            with torch.no_grad():
                output = model(
                    x=data.x,
                    edge_index=data.edge_index,
                    batch=data.batch)

            output = torch.sigmoid(output)
            _, predicted_labels = output.max(dim=1)

            micro, macro = __accuracy_aux(
                node_labels=data.node_labels,
                predicted_labels=predicted_labels,
                batch=data.batch, device=device)

            train_micro_avg += micro.cpu().numpy()
            train_macro_avg += macro.cpu().numpy()
            n_nodes += data.num_nodes
            n_graphs += data.num_graphs

        train_micro_avg = train_micro_avg / n_nodes
        train_macro_avg = train_macro_avg / n_graphs

    # ----- /TRAIN ------

    # ----- TEST 1 ------
    test1_micro_avg = 0.
    test1_macro_avg = 0.
    test1_loss = []
    test1_avg_loss = 0.

    if test_data1 is not None:
        n_nodes = 0
        n_graphs = 0
        for data in test_data1:
            data = data.to(device)

            with torch.no_grad():
                output = model(
                    x=data.x,
                    edge_index=data.edge_index,
                    batch=data.batch)

            loss = __loss_aux(
                output=output,
                loss=criterion,
                data=data,
                binary_prediction=binary_prediction)

            test1_loss.append(loss.detach().cpu().numpy())

            output = torch.sigmoid(output)
            _, predicted_labels = output.max(dim=1)

            micro, macro = __accuracy_aux(
                node_labels=data.node_labels,
                predicted_labels=predicted_labels,
                batch=data.batch, device=device)

            test1_micro_avg += micro.cpu().numpy()
            test1_macro_avg += macro.cpu().numpy()
            n_nodes += data.num_nodes
            n_graphs += data.num_graphs

        test1_avg_loss = np.mean(test1_loss)

        test1_micro_avg = test1_micro_avg / n_nodes
        test1_macro_avg = test1_macro_avg / n_graphs

    # ----- /TEST 1 ------

    # ----- TEST 2 ------
    test2_micro_avg = 0.
    test2_macro_avg = 0.
    test2_loss = []
    test2_avg_loss = 0.

    if test_data2 is not None:
        n_nodes = 0
        n_graphs = 0
        for data in test_data2:
            data = data.to(device)

            with torch.no_grad():
                output = model(
                    x=data.x,
                    edge_index=data.edge_index,
                    batch=data.batch)

            loss = __loss_aux(
                output=output,
                loss=criterion,
                data=data,
                binary_prediction=binary_prediction)

            test2_loss.append(loss.detach().cpu().numpy())

            output = torch.sigmoid(output)
            _, predicted_labels = output.max(dim=1)

            micro, macro = __accuracy_aux(
                node_labels=data.node_labels,
                predicted_labels=predicted_labels,
                batch=data.batch, device=device)

            test2_micro_avg += micro.cpu().numpy()
            test2_macro_avg += macro.cpu().numpy()
            n_nodes += data.num_nodes
            n_graphs += data.num_graphs

        test2_avg_loss = np.mean(test2_loss)

        test2_micro_avg = test2_micro_avg / n_nodes
        test2_macro_avg = test2_macro_avg / n_graphs

    # ----- /TEST 2 ------

    print(
        f"Train accuracy: micro: {train_micro_avg}\tmacro: {train_macro_avg}")
    print(f"Test1 loss: {test1_avg_loss}")
    print(f"Test2 loss: {test2_avg_loss}")
    print(f"Test accuracy: micro: {test1_micro_avg}\tmacro: {test1_macro_avg}")
    print(f"Test accuracy: micro: {test2_micro_avg}\tmacro: {test2_macro_avg}")

    return (train_micro_avg, train_macro_avg), \
        (test1_avg_loss, test1_micro_avg, test1_macro_avg), \
        (test2_avg_loss, test2_micro_avg, test2_macro_avg)


def seed_everything(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def main(
        args,
        manual,
        train_data=None,
        test1_data=None,
        test2_data=None,
        n_classes=None,
        save_model=True,
        load_model=None,
        train_model=False,
        activation=None):
    # set up seeds and gpu device
    seed_everything(args.seed)

    if torch.cuda.is_available():
        device = torch.device("cuda:" + str(args.device))
    else:
        device = torch.device("cpu")

    if not manual:
        raise NotImplementedError()
    else:
        assert train_data is not None
        assert test1_data is not None
        assert test2_data is not None
        assert n_classes is not None

        train_graphs = train_data
        test_graphs1 = test1_data
        test_graphs2 = test2_data

        if args.task_type == "node":
            num_classes = n_classes
        else:
            raise NotImplementedError()

    train_loader = DataLoader(
        train_graphs,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=0)
    test1_loader = DataLoader(
        test_graphs1,
        batch_size=512,
        pin_memory=True,
        num_workers=0)
    test2_loader = DataLoader(
        test_graphs2,
        batch_size=512,
        pin_memory=True,
        num_workers=0)

    # Model selection based on args
    if args.network == "acgnn":
        _model = ACGNN
    elif args.network == "acrgnn":
        _model = ACRGNN
    elif args.network == "gin":
        _model = GIN
    else:
        raise ValueError()

    model = _model(
        input_dim=train_graphs[0].num_features,
        hidden_dim=args.hidden_dim,
        output_dim=num_classes,
        num_layers=args.num_layers,
        aggregate_type=args.aggregate,
        readout_type=args.readout,
        combine_type=args.combine,
        combine_layers=args.combine_layers,
        num_mlp_layers=args.num_mlp_layers,
        task=args.task_type,
        activation=activation)

    if load_model is not None:
        print("Loading Model from", load_model)
        model.load_state_dict(torch.load(load_model))
    model = model.to(device)

    criterion = nn.BCEWithLogitsLoss(reduction='mean')
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

    if not args.filename == "":
        with open(args.filename, 'w') as f:
            f.write(
                "train_loss,test1_loss,test2_loss,train_micro,train_macro,test1_micro,test1_macro,test2_micro,test2_macro\n")
        with open(args.filename + ".train", 'w') as f:
            f.write("train_loss\n")
        with open(args.filename + ".test", 'w') as f:
            f.write("test1_loss,test2_loss\n")

    # If training is requested, run training loop
    if train_model:
        for epoch in range(1, args.epochs + 1):
            print(f"Epoch {epoch}/{args.epochs}")

            avg_loss, loss_iter = train(
                model=model,
                device=device,
                training_data=train_loader,
                optimizer=optimizer,
                criterion=criterion,
                scheduler=scheduler,
                binary_prediction=True)

            (train_micro, train_macro), (test1_loss, test1_micro, test1_macro), (test2_loss, test2_micro, test2_macro) = test(
                model=model, device=device, train_data=train_loader, test_data1=test1_loader, test_data2=test2_loader, epoch=epoch, criterion=criterion)

            file_line = f"{avg_loss: .10f}, {test1_loss: .10f}, {test2_loss: .10f}, {train_micro: .8f}, {train_macro: .8f}, {test1_micro: .8f}, {test1_macro: .8f}, {test2_micro: .8f}, {test2_macro: .8f}"

            if not args.filename == "":
                with open(args.filename, 'a') as f:
                    f.write(file_line + "\n")
                with open(args.filename + ".train", 'a') as f:
                    for l in loss_iter:
                        f.write(f"{l: .15f}\n")
                with open(args.filename + ".test", 'a') as f:
                    f.write(f"{test1_loss: .15f}, {test2_loss: .15f}\n")
    else:
        # If no training is requested, evaluate the loaded model once
        (train_micro, train_macro), (test1_loss, test1_micro, test1_macro), (test2_loss, test2_micro, test2_macro) = test(
            model=model, device=device, train_data=train_loader, test_data1=test1_loader, test_data2=test2_loader, epoch=-1, criterion=criterion)
        file_line = f" {-1: .8f}, {test1_loss: .10f}, {test2_loss: .10f}, {train_micro: .8f}, {train_macro: .8f}, {test1_micro: .8f}, {test1_macro: .8f}, {test2_micro: .8f}, {test2_macro: .8f}"
        if not args.filename == "":
            with open(args.filename, 'a') as f:
                f.write(file_line + "\n")

    # ***** DYNAMIC QUANTIZATION BLOCK *****
    if args.quantize is not None:
        print("Applying post-hoc dynamic quantization...")
        print("pytorch")
        model.eval()
        print("check LSTM",any(isinstance(m, nn.LSTM) for m in model.modules()))
        with open("model_quantization_summary.txt", "a") as f:
            f.write("\n========\n")
            full_model_name = load_model if load_model is not None else args.network
            f.write(full_model_name + "\n")
            checker=any(isinstance(m, nn.LSTM) for m in model.modules())
            f.write("check LSTM:" + str(checker) + "\n")
            f.write("Model before quantization:\n")
            f.write(str(model) + "\n\n")

            print("Model before quantization:")
            print(model)

            quantized_model = torch.quantization.quantize_dynamic(
                model,
                {nn.Linear},  # Add more layer types if needed
                dtype=torch.qint8
            )

            f.write("Model after quantization:\n")
            f.write(str(quantized_model) + "\n")

            print("Model after quantization:")
            print(quantized_model)
        with open("quantization_results.txt", "a") as f:
            f.write("\n========\n")
            full_model_name = load_model if load_model is not None else args.network
            f.write(full_model_name + "\n")
            f.write("\n==== New Quantization Run ====\n")
            for name, module in quantized_model.named_modules():
                if isinstance(module, torch.nn.quantized.Linear):
                    #torch.set_printoptions(threshold=float('inf'), sci_mode=False)
                    q_weight = module.weight()  # Returns a quantized tensor
                    if q_weight.is_quantized:
                        print("quantized:", name)
                        raw_int8_values = q_weight.int_repr()
                        f.write("\nquantized tensor:\n")
                        f.write(f"{raw_int8_values}\n")
                        print("qua",raw_int8_values)
                        scale = q_weight.q_scale()
                        zero_point = q_weight.q_zero_point()
                        result = (
                            f"Module: {name} | Weight scale: {scale} | "
                            f"Zero point: {zero_point} | Weight shape: {q_weight.size()}\n")
                        print(result.strip())  # Also print to console
                        f.write(result)
                        f.write("\ndequantized tensor:\n")
                        float_tensor = q_weight.dequantize()
                        f.write(f"{float_tensor}\n")
                        print("dequa",float_tensor)
                    else:
                        print("error", name)

        size_original = size_of_model(model)
        size_quantized = size_of_model(quantized_model)
        print("Size of the original model:", size_original)
        print("Size of the quantized model:", size_quantized)
        print("{0:.2f} times smaller".format(size_original/size_quantized))
        print("Time evaluation of the quantized model...")
        time_original = time_model_evaluation(model, device, train_loader, test1_loader, test2_loader, criterion)

        time_quantized = time_model_evaluation(quantized_model, device, train_loader, test1_loader, test2_loader, criterion)
        print(time_quantized)
        print("Testing the quantized model...")
        (q_train_micro, q_train_macro), (q_test1_loss, q_test1_micro, q_test1_macro), (q_test2_loss, q_test2_micro, q_test2_macro) = test(
            model=quantized_model, device=device, train_data=train_loader, test_data1=test1_loader, test_data2=test2_loader, epoch=-1, criterion=criterion)

        full_model_name = load_model if load_model is not None else args.network
        # Create a CSV line with your results
        csv_line = f"{full_model_name},{size_original},{size_quantized},{time_original: .1f},{time_quantized: .1f},{q_test1_loss: .10f},{q_test2_loss: .10f},{q_train_micro: .8f},{q_train_macro: .8f},{q_test1_micro: .8f},{q_test1_macro: .8f},{q_test2_micro: .8f},{q_test2_macro: .8f}\n"
        csv_filename = f"for_analysis/new_act_functions/results_synthetic/dymanic_quantized_results_size_time_pytorch_acrgnn_{ACTIVATION}.csv"

        # Check if file exists; if not, write the header, then append the new results.
        file_exists = os.path.exists(csv_filename)
        with open(csv_filename, 'a') as f:
            if not file_exists:
                f.write("model_name,size_original,size_quantized,time_original,time_quantized,test1_loss,test2_loss,train_micro,train_macro,test1_micro,test1_macro,test2_micro,test2_macro\n")
            f.write(csv_line)

        quantized_model_path = load_model.replace(".pth", "-quantized.pth")
        torch.save(quantized_model.state_dict(), quantized_model_path)
        print("Quantized model saved to:", quantized_model_path)

        
    else:
        if save_model is not None:
            torch.save(model.state_dict(), save_model)
    # ***** END OF DYNAMIC QUANTIZATION BLOCK *****

    

    return file_line + ","

if __name__ == '__main__':
    # agg, read, comb
    _networks = [
        {"add": "S"}, {"add": "S"}, {"simple": "T"}
    ]

    # Options: "relu", "relu6", "trrelu", "gelu", "sigmoid", "silu", "softplus", "elu". Is "relu" by default
    ACTIVATION = "elu"
    h = 64

    file_path = "."
    data_path = "data"
    extra_name = f"results_synthetic/acrgnn_{ACTIVATION}/"
    print("Start running")
    print("Current working directory of the main file:", os.getcwd())
    data_dir = "datasets"
    for key in ["p1", "p2", "p3"]:
    #key = "p1"
        for enum, _set in enumerate([
            [(f"{data_dir}/{key}/train-random-erdos-5000-40-50",
                f"{data_dir}/{key}/test-random-erdos-500-40-50",
                f"{data_dir}/{key}/test-random-erdos-500-51-60")
                ],
        ]):
            for index, (_train, _test1, _test2) in enumerate(_set):
                _train_graphs, (_, _, _n_node_labels) = load_data(
                    dataset=f"{file_path}/{data_path}/{_train}.txt",
                    degree_as_node_label=False)
                _test_graphs, _ = load_data(
                    dataset=f"{file_path}/{data_path}/{_test1}.txt",
                    degree_as_node_label=False)
                _test_graphs2, _ = load_data(
                    dataset=f"{file_path}/{data_path}/{_test2}.txt",
                    degree_as_node_label=False)
                _net_class = "acrgnn"
                filename = f"{file_path}/logging/{extra_name}{key}-{enum}-{index}.mix"
                _time_file =f"{file_path}/logging/{extra_name}{key}-{enum}-{index}time.time"
                (_agg, _agg_abr) = list(_networks[0].items())[0]
                (_read, _read_abr) = list(_networks[1].items())[0]
                (_comb, _comb_abr) = list(_networks[2].items())[0]
                for comb_layers in [1]:  #[0,1,2]: #in the paper we have comb_layers = 1, in original code [0,1,2]
                    #comb_layers = 1
                    for l in range(1, 11):
                        #l = 4
                        run_filename = f"{file_path}/logging/{extra_name}{key}-{enum}-{index}-{_net_class}-agg{_agg_abr}-read{_read_abr}-comb{_comb_abr}-cl{comb_layers}-L{l}.log"
                        _args = argument_parser().parse_args(
                            [
                                f"--readout={_read}",
                                f"--aggregate={_agg}",
                                f"--combine={_comb}",
                                f"--network={_net_class}",
                                f"--filename={run_filename}",
                                "--epochs=20",
                                f"--batch_size=128",
                                f"--hidden_dim={h}",
                                f"--num_layers={l}",
                                f"--combine_layers={comb_layers}",
                                f"--num_mlp_layers=2",
                                "--device=0",

                            ])
                        _args.quantize = True # True when you want to quantize the model None while training
                        start_time = time.time()
                        line = main(
                            _args,
                            manual=True,
                            train_data=_train_graphs,
                            test1_data=_test_graphs,
                            test2_data=_test_graphs2,
                            n_classes=_n_node_labels,
                            load_model=f"{file_path}/saved_models/{extra_name}{key}/MODEL-{_net_class}-0-agg{_agg_abr}-read{_read_abr}-comb{_comb_abr}-cl{comb_layers}-L{l}-H{h}.pth",
                            train_model=False,  # We're loading a trained model
                            #save_model=f"{file_path}/saved_models/{extra_name}{key}/MODEL-{_net_class}-{enum}-agg{_agg_abr}-read{_read_abr}-comb{_comb_abr}-cl{comb_layers}-L{l}-H{h}.pth",
                            #train_model=True,       
                            activation=ACTIVATION, #Options: "relu", "relu6", "trrelu", "gelu", "sigmoid", "silu", "softplus", "elu"
                        )
                        end_time = time.time()
                        print(
                            f"Time taken for {l} layers: {end_time - start_time} seconds")
                        
                        with open(filename, 'a') as f:
                            f.write(line)
                        with open(_time_file, 'a') as time_file:
                            time_file.write(f"Time taken for {l} layers: {end_time - start_time} seconds\n")    
                    with open(filename, 'a') as f:
                        f.write("\n")
                    
