# =======================================================
# Trabajo 2 - Algoritmo de Grover
# Archivo: trabajo2-GroverAlgorithm.py
# Autor: Mauricio David Correa Hernández.
# Fecha: 21 de abril de 2025.
# Materia: Computación Cuántica I.
# Descripción: Simulación del algoritmo de Grover.
# =======================================================

# =======================================================
# 0. IMPORTACIONES
# =======================================================
import argparse
import numpy as np
from enum import Enum
from functools import reduce
from fractions import Fraction

# =======================================================
# 1. DEFINICIONES BÁSICAS
# =======================================================
# -------------------------------------------------------
# Definimos las puertas cuánticas
# -------------------------------------------------------
class QuantumGate(Enum):
    H = (1 / np.sqrt(2)) * np.array([[1, 1], [1, -1]], dtype=complex)

# -------------------------------------------------------
# Definimos los estados de los qubits
# -------------------------------------------------------
class QubitState(Enum):
    KET_0 = "ket_0"
    KET_1 = "ket_1"
    KET_MINUS = "ket_minus"

    @property
    def value_array(self):
        if self == QubitState.KET_0:
            return np.array([[1], [0]], dtype=complex)
        elif self == QubitState.KET_1:
            return np.array([[0], [1]], dtype=complex)
        elif self == QubitState.KET_MINUS:
            return (1 / np.sqrt(2)) * (np.array([[1], [0]], dtype=complex) - np.array([[0], [1]], dtype=complex))

# =======================================================
# 2. FUNCIONES AUXILIARES
# =======================================================
# =======================================================
# Funciones para el producto tensorial y matrices
# =======================================================

# -------------------------------------------------------
# Producto tensorial entre varios vectores
# -------------------------------------------------------
def tensor_product(kets):
    return reduce(np.kron, kets)

# -------------------------------------------------------
# Producto de matrices.
# -------------------------------------------------------
def producto_matricial(matrices):
    resultado = matrices[0]
    for matriz in matrices[1:]:
        resultado = np.dot(resultado, matriz)
    return resultado

# =======================================================
# Funciones para la representación de números
# =======================================================
# -------------------------------------------------------
# Función para convertir un número a superíndice
# -------------------------------------------------------
def to_superindice(n):
    super_index_map = {
        '0': '⁰', '1': '¹', '2': '²', '3': '³', '4': '⁴',
        '5': '⁵', '6': '⁶', '7': '⁷', '8': '⁸', '9': '⁹',
        '10': '¹⁰', '11': '¹¹', '12': '¹²', '13': '¹³', '14': '¹⁴', 
        '15': '¹⁵', '16': '¹⁶', '17': '¹⁷', '18': '¹⁸', '19': '¹⁹',
        '20': '²⁰', '21': '²¹', '22': '²²', '23': '²³', '24': '²⁴',
        '25': '²⁵', '26': '²⁶', '27': '²⁷', '28': '²⁸', '29': '²⁹',
        '30': '³⁰', '31': '³¹', '32': '³²', '33': '³³', '34': '³⁴',
        '35': '³⁵', '36': '³⁶', '37': '³⁷', '38': '³⁸', '39': '³⁹',
        '40': '⁴⁰', '41': '⁴¹', '42': '⁴²', '43': '⁴³', '44': '⁴⁴',
        '45': '⁴⁵', '46': '⁴⁶', '47': '⁴⁷', '48': '⁴⁸', '49': '⁴⁹',
        '50': '⁵⁰'
    }
    return ''.join(super_index_map[d] for d in str(n))

# -------------------------------------------------------
# Función para mostrar las fases y amplitudes de los estados
# -------------------------------------------------------

def mostrar_fases_estado(estado, n_qubits):
    print(f"\nESTADO FINAL PARA {n_qubits} QUBITS (PROBABILIDADES)...\n")

    # Anchuras de columna.
    ancho_estado = n_qubits + 30
    ancho_amplitud = 15
    ancho_fase = 8

    # Imprime encabezado de la tabla
    print(f"|{'-' * ancho_estado}|{'-' * ancho_amplitud}|{'-' * ancho_fase}|")
    print(f"| {'ESTADO'.ljust(ancho_estado - 1)}| {'AMPLITUD'.ljust(ancho_amplitud - 1)}| {'FASE'.ljust(ancho_fase - 1)}|")
    print(f"|{'-' * ancho_estado}|{'-' * ancho_amplitud}|{'-' * ancho_fase}|")

    # Iterar sobre los estados cuánticos
    for i, amp_lista in enumerate(estado):
        amp = amp_lista[0]
        if np.abs(amp) > 1e-10:
            binario = format(i, f'0{n_qubits}b')
            amplitud = np.abs(amp).item()
            fase_rad = np.angle(amp).item()
            fraccion_pi = Fraction(fase_rad / np.pi).limit_denominator(16)

            if fraccion_pi.denominator == 1:
                fase_str = f"{fraccion_pi.numerator}π"
            else:
                fase_str = f"{fraccion_pi.numerator}/{fraccion_pi.denominator}π"

            # Formateo de salida alineada
            estado_str = f"|{binario}⟩ = |{i}⟩: {amp.real: .5f}{' +' if amp.imag >= 0 else '-'}{abs(amp.imag): .5f}j"
            print(f"| {estado_str.ljust(ancho_estado - 1)}| {amplitud: .5f}".ljust(ancho_amplitud + 10) + f"      | {fase_str.ljust(ancho_fase - 1)}|")

    # Imprime línea final
    print(f"|{'-' * ancho_estado}|{'-' * ancho_amplitud}|{'-' * ancho_fase}|")


# =======================================================
# 3. ALGORITMOS PARA EL ALGORITMO DE GROVER
# =======================================================
# -------------------------------------------------------
# Hadamard para n qubits
# -------------------------------------------------------
def hadamard_n(n):
    return reduce(np.kron, [QuantumGate.H.value] * n)

# -------------------------------------------------------
# Crea el estado inicial |0...0⟩
# -------------------------------------------------------
def initial_state(n_qubits):
    matrices = []
    input_state = tensor_product([QubitState.KET_0.value_array] * n_qubits)
    hadamard_todos = hadamard_n(n_qubits)
    matrices.append(hadamard_todos)
    matrices.append(input_state)
    resultado_producto_matricial = producto_matricial(matrices)
    return resultado_producto_matricial

# -------------------------------------------------------
# Creamos la función oráculo f: {0,1}^n → {0,1}
# Devuelve 1 solo para el valor objetivo, por ejemplo x = 5
# -------------------------------------------------------
def f(x, objetivo):
    return 1 if x == objetivo else 0 

# -------------------------------------------------------
# Construye el oráculo U_f usando la función f y
# el estado ancilla |−⟩
# U_f |x⟩|−⟩ = |x⟩|−⟩ si f(x) = 0
# U_f |x⟩|−⟩ = -|x⟩|-⟩ si f(x) = 1
# -------------------------------------------------------
def Uf_matrix(n_qubits, f):
    size = 2**n_qubits
    Uf = np.eye(size, dtype=complex)
    for i in range(size):
        if f(i):
            Uf[i, i] = -1
    return Uf

# -------------------------------------------------------
# Construye la matriz de difusión de Grover (inversión en el promedio)
# D = 2|s⟩⟨s| - I
# -------------------------------------------------------
def operador_difusion(n_qubits, estado):
    N = 2**n_qubits
    # Calculamos D = 2|ψ⟩⟨ψ| - I
    D = 2 * producto_matricial([estado, estado.T]) - np.identity(N)
    return D

# =======================================================
# 4. ALGORITMO DE GROVER - EJECUCIÓN
# =======================================================
# -------------------------------------------------------
# Algoritmo de Grover completo
# Aplica Uf y D iterativamente al estado inicial
# -------------------------------------------------------
def algoritmo_grover(n_qubits, f, objetivo, iteraciones=None):
    print(f"Ejecutando Grover con \'{n_qubits} qubits\' buscando el objetivo :: En decimal: {objetivo} ==> En binario: {format(objetivo, f'0{n_qubits}b')}")
    N = 2**n_qubits
    # Calcula el número óptimo de iteraciones
    if iteraciones is None:
        iteraciones = int(np.floor((np.pi / 4) * np.sqrt(N)))

    # Estado inicial |s⟩
    estado = initial_state(n_qubits)
    print(f"\nESTADO INICIAL...")
    print(f"H⊗ {to_superindice(n_qubits)}⊗ |0⟩⊗ {to_superindice(n_qubits)} =\n {estado}")
    
    # Construye operadores Uf (oráculo) y D (difusión)
    Uf = Uf_matrix(n_qubits, lambda x: f(x, objetivo))
    print(f"\nMATRIZ UNITARIA U_f:\n {Uf}")
    
    D = operador_difusion(n_qubits, estado)
    print(f"\nMATRIZ UNITARIA D:\n {D}")

    # Aplica las iteraciones de Grover
    for _ in range(iteraciones):
        estado = producto_matricial([Uf, estado])
        estado = producto_matricial([D, estado])

    return estado  # Devuelve el estado final


# =======================================================
# FUNCIÓN PRINCIPAL
# =======================================================
def main():
    parser = argparse.ArgumentParser(
        description="Simulación del algoritmo de Grover"
    )
    
    parser.add_argument(
        "--qubits",
        type=int,
        required=True,
        help="Número de qubits utilizados en la simulación"
    )

    parser.add_argument(
        "--objetivo",
        type=int,
        required=True,
        help="Índice objetivo a buscar (entero entre 0 y 2^n - 1)"
    )

    # Parsear los argumentos
    args = parser.parse_args()
    
    # Ejecutamos el algoritmo de Grover
    estado_final = algoritmo_grover(args.qubits, f, args.objetivo)
    
    # Muestra la probabilidad de medir cada estado
    mostrar_fases_estado(estado_final, args.qubits)

# Forma de ejecución:   'python3 nombre_archivo.py --qubits Numero_de_qubits --objetivo Valor_a_buscar_en_decimal'
# Ejemplo de ejecución: 'python3 trabajo2-GroverAlgorithm.py --qubits 3 --objetivo 5'
if __name__ == "__main__":
    main()