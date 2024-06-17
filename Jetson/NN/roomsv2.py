import math
import re

# Definicion sala cuadrilatero

class Quadrilateral:
    """
    Quadrilateral class defines the main elements of a quadrilateral,
    being a, b, c, d the lengths of each of the four sides, and alpha,
    beta, gamma and delta being the angles of the corners.
    """

    def __init__(self, a, b, c, d, alpha, beta, gamma, delta):
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta


class Room(Quadrilateral):
    def __init__(self, a, b, c, d, alpha, beta, gamma, delta, height):
        super().__init__(a, b, c, d, alpha, beta, gamma, delta)
        self.height = height

        self.vector = []
        self.set_vector()

    def set_vector(self):
        self.vector = [round(self.a), round(self.b), round(self.c), round(self.d),
                       round(self.alpha), round(self.beta), round(self.gamma), round(self.delta),
                       round(self.height)]

    def return_vector(self):
        return self.vector


class UTSRoom(Room):
    def __init__(self, a, b, c, d, alpha, beta, gamma, delta, height, grid_center, rt60):
        super().__init__(a, b, c, d, alpha, beta, gamma, delta, height)
        self.grid_center = grid_center
        self.rt60 = rt60

    def get_m_l_position(self, rir_folder):
        
        print(f'Extracting m_l positions from {rir_folder}.')\
        
        pattern = r'mic\[(.*?)\]_spk\[(.*?)\]'
        
        xl = 0
        yl = 0
        zl = 0
        xm = 0
        ym = 0
        zm = 0
        
        match = re.search(pattern, rir_folder)
        if match:
            mic_vector = list(map(int, match.group(1).split(',')))
            spk_vector = list(map(int, match.group(2).split(',')))
            xl = spk_vector[0]
            yl = spk_vector[1]
            zl = spk_vector[2]
            xm = mic_vector[0]
            ym = mic_vector[1]
            zm = mic_vector[2]
        else:
            print("It is not possible to extract m_l positions")

        return [round(xl), round(yl), round(zl), round(xm), round(ym), round(zm), self.rt60]

    def return_embedding(self, rir_folder):
        lis_mic_vector = self.get_m_l_position(rir_folder)
        room_vector = self.return_vector()
        return room_vector[:8] + lis_mic_vector[:6] + [room_vector[8]] + [lis_mic_vector[6]]


def return_room(emb):
    name = None
    if emb[0] == 490:
        name = 'Anechoic'
    if emb[0] == 355:
        name = 'Small'
    if emb[0] == 736:
        name = 'Medium'
    if emb[0] == 994:
        name = 'Large'
    if emb[0] == 600:
        name = 'Box'

    return name


if __name__ == "__main__":

    Small_Room = UTSRoom(310, 450, 310, 450, 90, 90, 90, 90, 249, [155, 225], 168)

    rir_folder = "newrir_mic[208, 70, 78]_spk[64, 318, 90]"

    vector = Small_Room.return_embedding(rir_folder)
    # Los datos que se introducen son = [Sala, Zona, Array, Speaker, Micrófono] según están definidos los archivos wav

    print("Embedding vector:")
    print(vector)
