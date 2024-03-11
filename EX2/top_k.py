import numpy as np


vectors = np.loadtxt("wordVectors.txt")
with open("vocab.txt") as file:
    vocab = [line.rstrip() for line in file]


def cosine_sim(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))


def sorted_cosine_sim_vec_w_idxs(chosen_vec, all_vecs):
    return sorted([(i, np.dot(chosen_vec, all_vecs[i]) /
                    (np.linalg.norm(chosen_vec) * np.linalg.norm(all_vecs[i])))
                     for i in range(len(all_vecs))], reverse=True, key=lambda x:x[1])


def most_similar(word, k):
    word_idx = vocab.index(word)
    word_vec = vectors[word_idx]
    top5 = [(vocab[i],round(j,3)) for i,j in sorted_cosine_sim_vec_w_idxs(word_vec, vectors)[1:1+k]]
    print(word+":")
    for neighbor, score in top5:
        print("\t"+neighbor+', '+str(score))
    print("")
    return top5


if __name__ == "__main__":
    words = ["dog", "england", "john", "explode", "office"]
    for word in words:
        most_similar(word, 5)

