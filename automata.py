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
        filas, cols = grid_old.shape

        for i in range(filas):
            for j in range(cols):
                estado = grid_old[i, j]
                vecinos = int(vecinos_total[i, j])
                s_local = float(self.sustrato[i, j])
                factor_s = 1.0 if s_local >= s_min else max(0.0, s_local / s_min)
                # Escalar la tabla por prob_div (0.5 reproduce valores base)
                p_tabla = PROB_POR_VECINOS.get(vecinos, 0.5)
                escala = prob_div / 0.5
                p_base = p_tabla * escala
                p_efectiva = p_base * factor_s

                if estado == 0:
                    if 0 < vecinos <= n0 and p_efectiva > 0 and self.rng.random() < p_efectiva:
                        nuevo_grid[i, j] = 2  # coloniza como crecimiento
                elif estado == 1:
                    # inhibicion espacial simple: siempre pasa a crecimiento
                    nuevo_grid[i, j] = 2
                elif estado == 2:
                    if vecinos <= n0 and p_efectiva > 0 and self.rng.random() < p_efectiva:
                        nuevo_grid[i, j] = 1  # activa division
                # sin muerte explicita

        self.grid = nuevo_grid

        return {
            "vacios": int((self.grid == 0).sum()),
            "division": int((self.grid == 1).sum()),
            "crecimiento": int((self.grid == 2).sum()),
        }

    def estado_actual(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.grid.copy(), self.sustrato.copy()
