import numpy as np
from typing import Dict, Tuple
from parametros import ParametrosCA


# Probabilidades base por numero de vecinos (modelo malo.py, escalables)
PROB_POR_VECINOS = {
    0: 0.5,
    1: 0.5,
    2: 0.25,
    3: 0.125,
    4: 0.05,
}


class MicrobialCA:
    """Automata celular bidimensional para crecimiento microbiano (sin muerte explicita)."""

    def __init__(self, params: ParametrosCA):
        self.params = params
        self.rng = np.random.default_rng(self.params.semilla)
        # Distribucion inicial basada en la concentracion microbiana inicial
        # Mapear x(0) (g/L) -> fraccion de ocupacion de la malla
        ref = max(1e-6, getattr(self.params, "referencia_concentracion", 10.0))
        frac = float(self.params.concentracion_microbiana_inicial) / float(ref)
        frac = max(0.0, min(1.0, frac))
        p_div = 0.5 * frac
        p_cre = 0.5 * frac
        p_empty = 1.0 - frac
        self.grid = self.rng.choice(
            [0, 1, 2],
            size=(self.params.filas, self.params.columnas),
            p=[p_empty, p_div, p_cre],
        ).astype(np.int8)
        self.sustrato = np.full(
            (self.params.filas, self.params.columnas),
            self.params.sustrato_inicial,
            dtype=np.float32,
        )

    def _vecinos_estado(self, estado: int) -> np.ndarray:
        """Cuenta vecinos de Moore en un estado dado usando desplazamientos circulares."""
        g = (self.grid == estado).astype(np.int8)
        total = np.zeros_like(g, dtype=np.int16)
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                if di == 0 and dj == 0:
                    continue
                total += np.roll(np.roll(g, di, axis=0), dj, axis=1)
        return total

    def _difundir_sustrato(self) -> None:
        s = self.sustrato
        kernel_mean = sum(
            np.roll(np.roll(s, di, axis=0), dj, axis=1)
            for di in (-1, 0, 1)
            for dj in (-1, 0, 1)
            if not (di == 0 and dj == 0)
        ) / 8.0
        self.sustrato = s + self.params.difusion * (kernel_mean - s)

    def _consumir_sustrato(self) -> None:
        self.sustrato -= (self.grid == 1) * self.params.consumo_division
        self.sustrato -= (self.grid == 2) * self.params.consumo_crecimiento
        np.clip(self.sustrato, 0, None, out=self.sustrato)

    def step(self, n0: int, prob_div: float) -> Dict[str, int]:
        """Ejecuta un paso temporal (reglas tipo malo.py, sin muerte)."""
        # 1) difundir y consumir sustrato
        self._difundir_sustrato()
        self._consumir_sustrato()

        vecinos_div = self._vecinos_estado(1)
        vecinos_cre = self._vecinos_estado(2)
        vecinos_total = vecinos_div + vecinos_cre

        grid_old = self.grid.copy()
        nuevo_grid = grid_old.copy()

        s_min = max(1e-6, self.params.sustrato_minimo)
        s = self.sustrato

        # factor de sustrato: 1 si hay suficiente, s/s_min si no
        factor_s = np.where(s >= s_min, 1.0, np.maximum(0.0, s / s_min))

        # Probabilidad base por vecinos (tabla escalable por prob_div)
        claves = np.array([0, 1, 2, 3, 4], dtype=np.int16)
        valores = np.array([PROB_POR_VECINOS[k] for k in range(5)], dtype=np.float64)
        indice = np.minimum(vecinos_total, 4)
        p_tabla = np.where(vecinos_total > 4, 0.5, valores[indice])

        escala = prob_div / 0.5
        p_efectiva = p_tabla * escala * factor_s

        # estado 1: siempre pasa a crecimiento (inhibicion espacial simple)
        nuevo_grid[grid_old == 1] = 2

        # Máscaras de transición (mutuamente excluyentes por estado)
        coloniza = (grid_old == 0) & (vecinos_total > 0) & (vecinos_total <= n0) & (p_efectiva > 0)
        divide = (grid_old == 2) & (vecinos_total <= n0) & (p_efectiva > 0)

        # Consumir RNG en orden fila-mayor igual que el loop original
        idx = np.flatnonzero(coloniza | divide)
        if idx.size:
            draws = self.rng.random(idx.size)
            p_flat = p_efectiva.ravel()[idx]
            ganan = idx[draws < p_flat]
            nuevo_grid.ravel()[ganan] = np.where(grid_old.ravel()[ganan] == 0, 2, 1)

        self.grid = nuevo_grid

        return {
            "vacios": int((self.grid == 0).sum()),
            "division": int((self.grid == 1).sum()),
            "crecimiento": int((self.grid == 2).sum()),
        }

    def estado_actual(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.grid.copy(), self.sustrato.copy()
