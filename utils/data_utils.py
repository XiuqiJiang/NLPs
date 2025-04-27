import numpy as np

def chopping(data, lim=48):
    """Removes peptide sequences longer than a certain thereshold from a list of sequences

    Args:
        data (list): List of sequences as strings 
        lim (int, optional): Length thereshold. Defaults to 48.

    Returns:
        list: List of sequences shorter than limit
    """
    chopped = []
    for seq in data:
        if len(seq) <= lim:
            chopped.append(seq)
    return chopped

def padding(data, begin_token='', end_token='-', lim=48):
    """Pads all sequences in the list to a certain length with an end token

    Args:
        data (list): List of sequences as strings 
        begin_token (str, optional): Character to pad the beginning of each sequence string. Defaults to ''.
        end_token (str, optional): Character to pad the end of each sequence string to reach the length limit. Defaults to '-'.
        lim (int, optional): Length thereshold. Defaults to 48.

    Returns:
        list: List of padded sequences 
    """
    padded = []
    for seq in data:
        temp = begin_token + seq + end_token * (lim - len(seq))
        padded.append(temp)
    return padded

def onehot_encoding(data, alphabet='ACDEFGHIKLMNPQRSTVWY'):
    """One-hot encoding of protein sequences

    Args:
        data (list): List of sequence strings to encode dim:(N, sequence length)
        alphabet (string, optional): The alphabet to use. Defaults to 'ACDEFGHIKLMNPQRSTVWY'.

    Returns:
        list: List of encoded sequences dim:(N, sequence length, alphabet length)
    """
    aa2hot = {}
    for i, aa in enumerate(alphabet):
        v = [0 for j in alphabet]
        v[i] = 1
        aa2hot[aa] = v

    onehot_encoded = []
    for seq in data:
        temp = []
        for aa in seq:
            temp.append(aa2hot[aa])
        onehot_encoded.append(temp)
    return onehot_encoded

def onehot_decoding(data, alphabet='ACDEFGHIKLMNPQRSTVWY'):
    """One-hot decoding of protein sequences

    Args:
        data (list):  List of one-hot encoded sequences dim:(N, sequence length, alphabet length)
        alphabet (_type_, optional): The alphabet to use. Defaults to 'ACDEFGHIKLMNPQRSTVWY'.

    Returns:
        list: List of decoded sequences as strings. dim:(N, sequence length) 
    """
    onehot_decoded = []
    for array in data:
        temp = ''
        for i, seq in enumerate(array):
            temp += alphabet[seq.index(max(seq))]
        onehot_decoded.append(temp)
    return onehot_decoded 