# Simulación del Algoritmo de Grover en Python

El Algoritmo de Grover es una técnica de búsqueda cuántica que ofrece una aceleración cuadrática respecto a los algoritmos clásicos. Su objetivo es encontrar una entrada $x$ para la cual una función $f(x)$ devuelve $1$, dentro de un espacio de búsqueda no estructurado. Esta implementación en Python simula el comportamiento de dicho algoritmo sin utilizar librerías específicas de computación cuántica como Qiskit, lo que permite observar el funcionamiento de los operadores cuánticos desde la matemática de matrices.

## 0. Enunciado del proyecto.
Implementar el algoritmo de Grover en un lenguaje de programación cualquiera utilizando operaciones matriciales. No se pueden utilizar librerías ni bibliotecas de funciones que implementen operaciones cuánticas. El programa debe recibir el tamaño del problema y la funcion oráculo como un parámetro, generar la matriz y generar el circuito que implemente el algoritmo de Grover para esa función oráculo. 

## 1. Fundamentos Cuánticos Implementados

Se parte de la definición de elementos fundamentales de la computación cuántica. Entre ellos se incluyen las compuertas cuánticas (como la compuertas de Hadamard) y los estados base de los qubits, tales como $|0\rangle$, $|1\rangle$, y $|-\rangle$. La representación matricial de estos elementos permite simular la evolución de estados cuánticos en Python usando álgebra lineal.

El estado $|-\rangle$ es especialmente importante porque se utiliza como qubit ancilla en la construcción del oráculo de Grover. Este se define como:

#### Estados base:
$$|0\rangle = 
\begin{bmatrix}
1 \\
0
\end{bmatrix}
,  \ \ \
|1\rangle = 
\begin{bmatrix}
0 \\
1
\end{bmatrix}
,  \ \ \
H|1\rangle = |-\rangle = \frac{1}{\sqrt{2}}(|0\rangle - |1\rangle)$$

#### Estado de un qubit:
$$
|\psi\rangle = \alpha|0\rangle - \beta|1\rangle
$$

#### Compuertas Hadamerd:
$$
H = 
\frac{1}{\sqrt{2}}
\begin{bmatrix}
1 & 1 \\
1 & -1
\end{bmatrix}
$$

### Código de implementación.

```python
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

# -------------------------------------------------------
# Hadamard para n qubits
# -------------------------------------------------------
def hadamard_n(n):
    return reduce(np.kron, [QuantumGate.H.value] * n)
```

## 2. Herramientas Matemáticas Auxiliares

Se han definido funciones auxiliares para facilitar operaciones típicas en el dominio cuántico:

- `tensor_product`: realiza el producto tensorial entre múltiples qubits.
- `producto_matricial`: permite aplicar secuencias de operaciones matriciales.
- `mostrar_fases_estado`: imprime en consola los estados base junto con sus amplitudes y fases, transformando las fases a una fracción de π para mayor claridad.

También se incluye una función `to_superindice` que permite representar los números de manera estética como superíndices, facilitando la visualización del número de qubits o iteraciones en notación compacta.

```python
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
```

## 3. Construcción del Estado Inicial

La función `initial_state(n_qubits)` construye el estado $|\psi_0\rangle$, que representa una superposición uniforme de todos los posibles estados del sistema:
$$|\psi_0\rangle = H^{\otimes n} |0\rangle^{\otimes n}$$
Esto se logra aplicando la puerta de Hadamard a cada uno de los qubits partiendo del estado base $|0...0\rangle$.

```python
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
```

## 4. Implementación del Oráculo Cuántico

El oráculo cuántico $U_f$ implementa una función booleana que devuelve 1 solo para el estado objetivo $x_0$. A nivel matricial, esto se representa como una matriz identidad con una inversión de fase (-1) únicamente en la posición correspondiente al estado objetivo:
$$
U_f |x\rangle = 
\begin{cases} 
\- |x\rangle, & \text{si } f(x) = 1 \\
\ \ |x\rangle, & \text{si } f(x) = 0 
\end{cases}
$$

La matriz se construye dinámicamente en la función `Uf_matrix(n_qubits, f)`, iterando sobre los posibles índices del estado.

```python
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
```

## 5. Matriz de Difusión (Difusor de Grover)

Una parte crítica del algoritmo de Grover es la **difusión**, que realiza una inversión sobre la media. Esto se conoce como el operador de Grover $D$, y se expresa como:
$$D = 2|s\rangle\langle s| - I$$
En este código, se construye esta matriz utilizando el estado inicial $|\psi_0\rangle$, y una matriz identidad $I$. Esta operación amplifica la amplitud del estado marcado mientras disminuye la de los demás, lo cual mejora la probabilidad de medir la respuesta correcta.

```python
# -------------------------------------------------------
# Construye la matriz de difusión de Grover (inversión en el promedio)
# D = 2|s⟩⟨s| - I
# -------------------------------------------------------
def operador_difusion(n_qubits, estado):
    N = 2**n_qubits
    # Calculamos D = 2|ψ⟩⟨ψ| - I
    D = 2 * producto_matricial([estado, estado.T]) - np.identity(N)
    return D
```

## 6. Medición de Resultados

Al final del proceso, se mide el estado resultante y se imprime una tabla que muestra:

- Los estados base $|x\rangle$
- Sus amplitudes complejas
- La fase asociada (en múltiplos de π)

Esto se realiza a través de la función `mostrar_fases_estado`, que convierte la amplitud y fase de cada estado a una forma comprensible y didáctica para el usuario.

```python
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
```

## 7. Algoritmo de Grover completo

El algoritmo de Grover es una técnica cuántica con una complejidad de $O(\sqrt{N})$, mejorando significativamente el rendimiento comparado con los algoritmos clásicos, cuya complejidad es $O(N)$.

En esta sección se implementa la **ejecución completa** del algoritmo de Grover. El proceso consta de las siguientes etapas:

Dado un número de qubits $n$, se quiere encontrar un elemento objetivo dentro de un espacio de búsqueda de tamaño $N = 2^n$. El objetivo está representado por un número decimal específico que se codifica en binario para definir la función booleana $f(x)$, utilizada por el **oráculo cuántico** $U_f$.


### Parámetros del algoritmo

La función `algoritmo_grover` recibe como argumentos:

- `n_qubits`: número de qubits en el sistema, que determina el espacio de búsqueda.
- `f`: función booleana $f(x)$, que retorna `1` si `x` es la solución, y `0` en caso contrario.
- `objetivo`: el elemento que se busca.
- `iteraciones` *(opcional)*: número de veces que se aplican los operadores $U_f$ (oráculo) y $D$ (difusión). Si no se especifica, se calcula el número óptimo como:

$$R = \left\lfloor \frac{\pi}{4} \sqrt{N} \right\rfloor$$

### Paso 1: Inicialización del estado

Se construye el estado inicial $|s\rangle$ aplicando compuertas Hadamard $H^{\otimes n}$ al estado base $|0\rangle^{\otimes n}$. Esto genera una **superposición uniforme**:

$$|s\rangle = \frac{1}{\sqrt{N}} \sum_{x=0}^{N-1} |x\rangle$$

Esta superposición representa la incertidumbre total sobre la posición del objetivo.

### Paso 2: Construcción de operadores

Se construyen dos matrices unitarias clave:

- **$ U_f $**: el **oráculo cuántico**, que invierte el signo del estado correspondiente al objetivo:

  $$U_f |x\rangle = (-1)^{f(x)} |x\rangle$$

- **$ D $**: el **operador de difusión**, también llamado inversión sobre la media, que amplifica la probabilidad del estado objetivo. Matemáticamente:

  $$D = 2|s\rangle\langle s| - I$$

  donde $I$ es la matriz identidad con N filas y columnas.


### Paso 3: Iteración del algoritmo

El núcleo del algoritmo consiste en aplicar repetidamente la combinación de los dos operadores:

1. Se aplica el oráculo $U_f$ al estado.
2. Se aplica el operador de difusión $D$.

Este proceso se repite $R$ veces, donde cada iteración incrementa la amplitud del estado objetivo y reduce la de los demás.

### Paso 4: Resultado

Después de las iteraciones, el estado final `estado` es devuelto por la función. Este estado tiene **alta probabilidad de colapsar en el estado objetivo** si se realizara una medición.

```python
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
        estado = np.dot(Uf, estado)  # Aplica el oráculo
        estado = np.dot(D, estado)   # Aplica la difusión

    return estado  # Devuelve el estado final
```

## 8. Función principal.

En esta sección se crea un objeto ArgumentParser para gestionar los argumentos del usuario

- Se define el argumento obligatorio `--qubits` de tipo entero que indica cuántos qubits se utilizarán en la simulación.
- Se define el argumento obligatorio `--objetivo` de tipo entero que indica el índice (en decimal) del estado objetivo a buscar.
- Se realiza el análisis de los argumentos proporcionados por el usuario
- Llama a la función `algoritmo_grover` con los argumentos proporcionados `f` es la función booleana que determina cuál es el objetivo correcto.
- Se muestran las amplitudes y probabilidades de todos los estados.

```python
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
```

## 9. Ejemplo de ejecución.

Para ejecutar el programa es de la siguiente manera.

### Uso.
```bash
python3 trabajo2-GroverAlgorithm.py --qubits <n> --objetivo <estado_decimal>
```

### Ejemplo.
```bash
python3 trabajo2-GroverAlgorithm.py --qubits 3 --objetivo 5
```

### Resultados de ejecución.
```bash
user@pc:~$ python3 trabajo2-GroverAlgorithm.py --qubits 3 --objetivo 5
Ejecutando Grover con '3 qubits' buscando el objetivo :: En decimal: 5 ==> En binario: 101

ESTADO INICIAL...
H⊗ ³⊗ |0⟩⊗ ³ =
 [[0.35355339+0.j]
 [0.35355339+0.j]
 [0.35355339+0.j]
 [0.35355339+0.j]
 [0.35355339+0.j]
 [0.35355339+0.j]
 [0.35355339+0.j]
 [0.35355339+0.j]]

MATRIZ UNITARIA U_f:
 [[ 1.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j]
 [ 0.+0.j  1.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j]
 [ 0.+0.j  0.+0.j  1.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j]
 [ 0.+0.j  0.+0.j  0.+0.j  1.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j]
 [ 0.+0.j  0.+0.j  0.+0.j  0.+0.j  1.+0.j  0.+0.j  0.+0.j  0.+0.j]
 [ 0.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j -1.+0.j  0.+0.j  0.+0.j]
 [ 0.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j  1.+0.j  0.+0.j]
 [ 0.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j  1.+0.j]]

MATRIZ UNITARIA D:
 [[-0.75+0.j  0.25+0.j  0.25+0.j  0.25+0.j  0.25+0.j  0.25+0.j  0.25+0.j
   0.25+0.j]
 [ 0.25+0.j -0.75+0.j  0.25+0.j  0.25+0.j  0.25+0.j  0.25+0.j  0.25+0.j
   0.25+0.j]
 [ 0.25+0.j  0.25+0.j -0.75+0.j  0.25+0.j  0.25+0.j  0.25+0.j  0.25+0.j
   0.25+0.j]
 [ 0.25+0.j  0.25+0.j  0.25+0.j -0.75+0.j  0.25+0.j  0.25+0.j  0.25+0.j
   0.25+0.j]
 [ 0.25+0.j  0.25+0.j  0.25+0.j  0.25+0.j -0.75+0.j  0.25+0.j  0.25+0.j
   0.25+0.j]
 [ 0.25+0.j  0.25+0.j  0.25+0.j  0.25+0.j  0.25+0.j -0.75+0.j  0.25+0.j
   0.25+0.j]
 [ 0.25+0.j  0.25+0.j  0.25+0.j  0.25+0.j  0.25+0.j  0.25+0.j -0.75+0.j
   0.25+0.j]
 [ 0.25+0.j  0.25+0.j  0.25+0.j  0.25+0.j  0.25+0.j  0.25+0.j  0.25+0.j
  -0.75+0.j]]

ESTADO FINAL PARA 3 QUBITS (PROBABILIDADES)...

|---------------------------------|---------------|--------|
| ESTADO                          | AMPLITUD      | FASE   |
|---------------------------------|---------------|--------|
| |000⟩ = |0⟩: -0.08839 + 0.00000j|  0.08839      | 1π     |
| |001⟩ = |1⟩: -0.08839 + 0.00000j|  0.08839      | 1π     |
| |010⟩ = |2⟩: -0.08839 + 0.00000j|  0.08839      | 1π     |
| |011⟩ = |3⟩: -0.08839 + 0.00000j|  0.08839      | 1π     |
| |100⟩ = |4⟩: -0.08839 + 0.00000j|  0.08839      | 1π     |
| |101⟩ = |5⟩:  0.97227 + 0.00000j|  0.97227      | 0π     |
| |110⟩ = |6⟩: -0.08839 + 0.00000j|  0.08839      | 1π     |
| |111⟩ = |7⟩: -0.08839 + 0.00000j|  0.08839      | 1π     |
|---------------------------------|---------------|--------|
```

## Conclusión

Este simulador del algoritmo de Grover, escrito íntegramente en Python usando matrices de NumPy, representa una implementación clara del poder de la computación cuántica sin recurrir a simuladores especializados. Al trabajar desde los principios matemáticos, se tiene una comprensión más profunda de los efectos de la transformación cuántica paso a paso, incluyendo la construcción del oráculo, la difusión y la evolución del sistema.

Además, permite experimentar con distintos tamaños de espacio de búsqueda (número de qubits) y distintos objetivos, brindando una herramienta educativa valiosa para quienes están aprendiendo los principios fundamentales de la computación cuántica.

# Referencias.

- [Grover's algorithm - IBM/Tutorial](https://learning.quantum.ibm.com/tutorial/grovers-algorithm)
- [Grover's algorithm - IBM/Course](https://learning.quantum.ibm.com/course/fundamentals-of-quantum-algorithms/grovers-algorithm)
- [quantum.ibm.com - IBM](https://quantum.ibm.com/)
- [Quantum Algorithm Zoo](https://quantumalgorithmzoo.org/)
- [Tutorial: Implementación del algoritmo de búsqueda de Grover en Q# - Microsoft](https://learn.microsoft.com/es-es/azure/quantum/tutorial-qdk-grovers-search?tabs=tabid-vscode#the-grovers-algorithm)
- [15 3 Implementing Grovers algorithm 23 mins - YouTube](https://www.youtube.com/watch?v=XumGT8Ed84Q&list=PLnhoxwUZN7-6hB2iWNhLrakuODLaxPTOG&index=65&ab_channel=AritraSarkar)
- [numpy.outer - Numpy](https://numpy.org/doc/2.1/reference/generated/numpy.outer.html)
- [Algoritmo de Grover - Wikipedia](https://es.wikipedia.org/wiki/Algoritmo_de_Grover)
- [Producto de Kronecker - Wikipedia](https://es.wikipedia.org/wiki/Producto_de_Kronecker)
