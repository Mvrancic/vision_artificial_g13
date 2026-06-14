import os
import sys

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from enfoque1_perspectiva import main as e1
from enfoque2_renovacion import main as e2
from enfoque3_reconstruccion import main as e3


MENU = """
============================================================
  TP Final - Vision Artificial aplicada a Diseno de Interiores
============================================================
  1) Visualizacion de muebles en perspectiva real (homografia)
  2) Renovacion inteligente: borrar mueble + inpainting
  3) Reconstructor de ambientes y planos 3D (profundidad)
  q) Salir
------------------------------------------------------------
  Fuente:  e=imagen del repo (default)   c=camara
============================================================
"""


def ask_source():
    src = input('Fuente [e/c] (enter=imagen): ').strip().lower()
    return src == 'c'


def main():
    while True:
        print(MENU)
        opt = input('Elegi una opcion: ').strip().lower()
        if opt == 'q':
            break
        elif opt == '1':
            e1.run(use_camera=ask_source())
        elif opt == '2':
            e2.run(use_camera=ask_source())
        elif opt == '3':
            e3.run(use_camera=ask_source())
        else:
            print('Opcion invalida.')


if __name__ == '__main__':
    main()
