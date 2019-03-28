import torch
import config


def main(corpus):
    calculate_similarity(corpus)


def calculate_similarity(corpus):
    type_atten_matrix = torch.load(config.DATA_ROOT + corpus + "type_atten.pt", map_location=torch.device('cpu'))
    idx2type = {}
    with open(config.DATA_ROOT + corpus + "type_set.txt", 'r') as f:
        lines = f.readlines()
        for l in lines:
            token = l[:-1].split(" ")
            idx2type[int(token[0])] = token[1]
    # sim = type_atten_matrix.matmul(type_atten_matrix.t())

    norm = type_atten_matrix.norm(dim=1).unsqueeze(1)

    sim = type_atten_matrix.matmul(type_atten_matrix.t()).div(
        norm.matmul(norm.t())) - torch.eye(norm.shape[0], dtype=torch.float)
    print(sim.shape)
    print(list(sim[0].detach().numpy()))
    for i, row in enumerate(sim):
        print(f"Type: {idx2type[i]}     Closest type: {idx2type[torch.argmax(row).item()]}")
        sims = zip(list(idx2type.values()), list(sim[i].detach().numpy()))
        print(f"{list(sims)}")
        print()


if __name__ == '__main__':
    main(config.WIKI)
