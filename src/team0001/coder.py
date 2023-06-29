_author__ = "Zhang, Haoling [zhanghaoling@genomics.cn]"


from numpy import array, zeros, arange, fromfile, packbits, unpackbits, expand_dims, concatenate, add
from numpy import log2, max, sum, ceil, where, uint8, uint64

from evaluation import DefaultCoder


class Coder(DefaultCoder):

    def __init__(self, team_id: str = "none"):
        """
        Initialize the image-DNA coder.

        :param team_id: team id provided by sponsor.
        :type team_id: str

        .. note::
            The competition process is automatically created.

            Thus,
            (1) Please do not add parameters other than "team_id".
                All parameters should be declared directly in this interface instead of being passed in as parameters.
                If a parameter depends on the input image, please assign its value in the "image_to_dna" interface.
            (2) Please do not divide "coder.py" into multiple script files.
                Only the script called "coder.py" will be automatically copied by
                the competition process to the competition script folder.

        """
        super().__init__(team_id=team_id)
        self.address, self.payload = 12, 128
        self.supplement, self.message_number = 0, 0

    def image_to_dna(self, input_image_path, need_logs=True):
        """
        Convert an image into a list of DNA sequences.

        :param input_image_path: path of the image to be encoded.
        :type input_image_path: str

        :param need_logs: print process logs if required.
        :type need_logs: bool

        :return: a list of DNA sequences.
        :rtype: list

        .. note::
            Each DNA sequence is suggested to carry its address information in the sequence list.
            Because the DNA sequence list obtained in DNA sequencing is inconsistent with the existing list.
        """
        if need_logs:
            print("Obtain binaries from file.")
        bits = unpackbits(expand_dims(fromfile(file=input_image_path, dtype=uint8), 1), axis=1).reshape(-1)
        if need_logs:
            print("%d bits are obtained." % len(bits))

        if len(bits) % (self.payload * 2) != 0:
            self.supplement = self.payload * 2 - len(bits) % (self.payload * 2)
            if need_logs:
                print("Supplement %d bits to make sure all payload lengths are same." % self.supplement)
            bits = concatenate((bits, zeros(shape=(self.supplement,), dtype=uint8)), axis=0)
        binary_messages = bits.reshape(len(bits) // (128 * 2), (128 * 2))
        self.message_number = len(binary_messages)

        if need_logs:
            print("Insert index for each binary message.")
        byte_number = ceil(log2(len(binary_messages)) / log2(256)).astype(int)
        mould = zeros(shape=(len(binary_messages), byte_number), dtype=uint8)
        integers = arange(len(binary_messages), dtype=int)
        for index in range(byte_number):
            mould[:, -1 - index] = integers % 256
            integers //= 256
        index_matrix = unpackbits(expand_dims(mould.reshape(-1), axis=1), axis=1)
        index_matrix = index_matrix.reshape(len(binary_messages), byte_number * 8)
        unused_locations = where(sum(index_matrix, axis=0) == 0)[0]
        start_location = max(unused_locations) + 1 if len(unused_locations) > 0 else 0
        index_matrix = index_matrix[:, start_location:]
        if self.address * 2 > len(index_matrix[0]):  # use the given address length.
            expanded_matrix = zeros(shape=(len(binary_messages), self.address * 2 - len(index_matrix[0])), dtype=uint8)
            index_matrix = concatenate((expanded_matrix, index_matrix), axis=1)
        elif self.address * 2 < len(index_matrix[0]):
            raise ValueError("The address length is too short to represent all addresses.")
        binary_messages = concatenate((index_matrix, binary_messages), axis=1)

        if need_logs:
            print("Encode binary messages based on the mapping scheme.")
        digit_set = 2 * binary_messages[:, :self.address + self.payload]
        digit_set += binary_messages[:, self.address + self.payload:]
        digit_set[digit_set == 0] = ord("A")
        digit_set[digit_set == 1] = ord("C")
        digit_set[digit_set == 2] = ord("G")
        digit_set[digit_set == 3] = ord("T")
        dna_sequences = []
        for digits in digit_set:
            dna_sequences.append(digits.tostring().decode("ascii"))
            if need_logs:
                self.monitor(len(dna_sequences), len(binary_messages))

        return dna_sequences

    def dna_to_image(self, dna_sequences, output_image_path, need_logs=True):
        """
        Convert a list of DNA sequences to an image.

        :param dna_sequences: a list of DNA sequences (obtained from DNA sequencing).
        :type dna_sequences: list

        :param output_image_path: path for storing image data.
        :type output_image_path: str

        :param need_logs: print process logs if required.
        :type need_logs: bool

        .. note::
           The order of the samples in this DNA sequence list input must be different from
           the order of the samples output by the "image_to_dna" interface.
        """
        if need_logs:
            print("Decode DNA sequences based on the mapping scheme.")
        binary_messages = zeros(shape=(self.message_number, 2 * (self.address + self.payload)), dtype=uint8)
        mapping = {"A": [0, 0], "C": [0, 1], "G": [1, 0], "T": [1, 1]}
        for index, dna_sequence in enumerate(dna_sequences):
            for nucleotide_index, nucleotide in enumerate(dna_sequence[:self.address + self.payload]):
                upper, lower = mapping[nucleotide]
                binary_messages[index, nucleotide_index] = upper
                binary_messages[index, nucleotide_index + self.address + self.payload] = lower
            if need_logs:
                self.monitor(index + 1, len(dna_sequences))
        binary_messages = array(binary_messages, dtype=uint8)

        if need_logs:
            print("Sort binary messages and convert them as bits.")
        index_matrix, binary_messages = binary_messages[:, :self.address * 2], binary_messages[:, self.address * 2:]
        message_number, byte_number = len(index_matrix), ceil(len(index_matrix[0]) / 8).astype(int)
        if len(index_matrix[0]) % 8 != 0:
            expanded_matrix = zeros(shape=(message_number, 8 * byte_number - len(index_matrix[0])), dtype=uint8)
            template = concatenate((expanded_matrix, index_matrix), axis=1)
        else:
            template = index_matrix
        mould = packbits(template.reshape(message_number * byte_number, 8), axis=1).reshape(message_number, byte_number)
        orders = zeros(shape=(message_number,), dtype=uint64)
        for index in range(byte_number):
            orders = add(orders, mould[:, -1 - index] * (256 ** index), out=orders,
                         casting="unsafe")  # make up according to the byte scale.
        sorted_binary_messages = zeros(shape=(self.message_number, 2 * self.payload), dtype=uint8)
        for index, order in enumerate(orders):
            if order < len(sorted_binary_messages):
                sorted_binary_messages[order] = binary_messages[index]
            if need_logs:
                self.monitor(index + 1, len(orders))

        bits = sorted_binary_messages.reshape(-1)
        bits = bits[:-self.supplement]
        if need_logs:
            print("%d bits are retrieved." % len(bits))

        if need_logs:
            print("Save bits to the file.")
        byte_array = packbits(bits.reshape(len(bits) // 8, 8), axis=1).reshape(-1)
        byte_array.tofile(file=output_image_path)
