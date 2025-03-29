# numpy para resolver o exercício 2
import numpy as np

# Função para ler os arquivos FASTA e extrair as sequências de DNA
def read_fasta(arq):
    seq = ''
    with open(arq) as f:
        f.readline()
        for line in f:
            seq += line.strip()
    return seq

# Extraindo e armazenando as sequências dos organismos
seq_hamster = read_fasta("hamster.fasta")
seq_horse = read_fasta("horse.fasta")
seq_rat = read_fasta("rat.fasta")

# 1. Comparação de Proximidade
# Função para comparação de proximidade por verificação simples
def comparar_prox(seq1, seq2):
    # m: número de atributos com valores iguais
    m = 0
    # p: número total de atributos
    p = len(seq1)

    i = 0
    i2 = 0

    # laço de repetição para contar quantos atributos com valores iguais tem
    while i < p:
        if seq1[i] == seq2[i2]:
            m = m + 1
        i+=1
        i2+=1

    # Fórmula da Aula 3
    proximidade = (p - m)/p
    return proximidade

print('Comparação da proximidade entre hamster e cavalo: ', comparar_prox(seq_hamster, seq_horse))
print('Comparação da proximidade entre hamster e rato: ', comparar_prox(seq_hamster, seq_rat))
print('Comparação da proximidade entre rato e cavalo: ', comparar_prox(seq_rat, seq_horse))

# 2. Contagem de aminoácidos e Cálculo de similaridade
# a) Conte a ocorrência de cada aminoácido nas sequências fornecidas e crie um vetor numérico representando essas ocorrências.
def vetor_aminoacido(seq):
    aminoacidos = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

    # dicionário p contagem dos aminoácidos
    contagem_aminoacidos = {aa: 0 for aa in aminoacidos}

    # contagem da ocorrência de aminoácidos na sequência
    for aa in seq:
        if aa in contagem_aminoacidos:
            contagem_aminoacidos[aa] = contagem_aminoacidos[aa] + 1

    # criar um vetor com a contagem de cada aminoácido
    vetor = [contagem_aminoacidos[aa] for aa in aminoacidos]

    return vetor

vetor_hamster = vetor_aminoacido(seq_hamster)
vetor_horse = vetor_aminoacido(seq_horse)
vetor_rat = vetor_aminoacido(seq_rat)

#b) Calcule as seguintes distâncias e similaridades entre os pares de organismos:
def dist_similaridade(seq1, seq2):
    vetor1 = np.array(seq1, dtype=np.int64)
    vetor2 = np.array(seq2, dtype=np.int64)

    # Distância Manhattan
    manhattan = np.sum(np.abs(vetor1-vetor2))
    print("A distância Manhattan é: ", manhattan)

    # Distância Euclidiana
    euclidiana = np.linalg.norm(vetor1 - vetor2)
    print("A distância Euclidiana é: ", euclidiana)

    #Distância Supremum
    supremum = np.max(np.abs(vetor1-vetor2))
    print("A distância Supremum é: ",supremum)

    #Similaridade de Cosseno
    p_escalar = np.dot(vetor1, vetor2)
    comprimento_v1 = np.linalg.norm(vetor1)
    comprimento_v2 = np.linalg.norm(vetor2)

    cosseno = p_escalar/(comprimento_v1 * comprimento_v2)

    print("A distância do cosseno é: ", cosseno)

print("Hamster vs Cavalo")
dist_similaridade(vetor_hamster, vetor_horse)
print("Hamster vs Rato")
dist_similaridade(vetor_hamster, vetor_rat)
print("Rato vs Cavalo")
dist_similaridade(vetor_rat, vetor_horse)
