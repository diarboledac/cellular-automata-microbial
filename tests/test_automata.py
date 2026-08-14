"""Tests para el automata celular microbiano (cellular-automata-microbial)."""

from __future__ import annotations

import types

import numpy as np
import pytest

from automata import PROB_POR_VECINOS, MicrobialCA
from parametros import ParametrosCA


def parametros_pequenos() -> ParametrosCA:
    p = ParametrosCA()
    p.filas = 40
    p.columnas = 40
    p.iteraciones = 10
    return p


# ---------------------------------------------------------------------------
# Referencia: implementacion en loop explicito (la "especificacion").
# El paso vectorizado debe producir resultados identicos con la misma semilla.
# ---------------------------------------------------------------------------

def step_referencia(ca: MicrobialCA, n0: int, prob_div: float) -> dict[str, int]:
    """Copia exacta de la logica original de MicrobialCA.step en loop puro."""
    ca._difundir_sustrato()
    ca._consumir_sustrato()

    vecinos_div = ca._vecinos_estado(1)
    vecinos_cre = ca._vecinos_estado(2)
    vecinos_total = vecinos_div + vecinos_cre

    grid_old = ca.grid.copy()
    nuevo_grid = grid_old.copy()

    s_min = max(1e-6, ca.params.sustrato_minimo)
    filas, cols = grid_old.shape

    for i in range(filas):
        for j in range(cols):
            estado = grid_old[i, j]
            vecinos = int(vecinos_total[i, j])
            s_local = float(ca.sustrato[i, j])
            factor_s = 1.0 if s_local >= s_min else max(0.0, s_local / s_min)
            p_tabla = PROB_POR_VECINOS.get(vecinos, 0.5)
            escala = prob_div / 0.5
            p_base = p_tabla * escala
            p_efectiva = p_base * factor_s

            if estado == 0:
                if 0 < vecinos <= n0 and p_efectiva > 0 and ca.rng.random() < p_efectiva:
                    nuevo_grid[i, j] = 2
            elif estado == 1:
                nuevo_grid[i, j] = 2
            elif estado == 2:
                if vecinos <= n0 and p_efectiva > 0 and ca.rng.random() < p_efectiva:
                    nuevo_grid[i, j] = 1

    ca.grid = nuevo_grid

    return {
        "vacios": int((ca.grid == 0).sum()),
        "division": int((ca.grid == 1).sum()),
        "crecimiento": int((ca.grid == 2).sum()),
    }


# ---------------------------------------------------------------------------
# Tests de comportamiento
# ---------------------------------------------------------------------------

def test_grid_inicial_shape_y_estados_validos():
    p = parametros_pequenos()
    ca = MicrobialCA(p)
    assert ca.grid.shape == (40, 40)
    assert set(np.unique(ca.grid)).issubset({0, 1, 2})


def test_grid_inicial_es_determinista_con_misma_semilla():
    p1 = parametros_pequenos()
    p2 = parametros_pequenos()
    ca1 = MicrobialCA(p1)
    ca2 = MicrobialCA(p2)
    assert np.array_equal(ca1.grid, ca2.grid)


def test_grid_inicial_con_distinta_semilla_difiere():
    p1 = parametros_pequenos()
    p2 = parametros_pequenos()
    p2.semilla = 1
    ca1 = MicrobialCA(p1)
    ca2 = MicrobialCA(p2)
    assert not np.array_equal(ca1.grid, ca2.grid)


def test_sustrato_inicial_uniforme():
    p = parametros_pequenos()
    ca = MicrobialCA(p)
    assert np.allclose(ca.sustrato, p.sustrato_inicial)


def test_step_devuelve_estados_que_suman_total():
    p = parametros_pequenos()
    ca = MicrobialCA(p)
    counts = ca.step(n0=3, prob_div=0.5)
    assert set(counts) == {"vacios", "division", "crecimiento"}
    assert counts["vacios"] + counts["division"] + counts["crecimiento"] == 40 * 40


def test_step_celula_en_division_siempre_pasa_a_crecimiento():
    p = parametros_pequenos()
    ca = MicrobialCA(p)
    ca.grid[:] = 0
    ca.grid[5, 5] = 1
    ca.step(n0=3, prob_div=0.5)
    assert ca.grid[5, 5] == 2


def test_celula_vacia_sin_vecinos_no_coloniza():
    p = parametros_pequenos()
    ca = MicrobialCA(p)
    ca.grid[:] = 0
    ca.grid[0, 0] = 1  # unica celula viva, fuera del radio de vecindad de [20,20]
    ca.step(n0=3, prob_div=0.5)
    assert ca.grid[20, 20] == 0


def test_sustrato_no_negativo_y_decrece_con_consumo():
    p = parametros_pequenos()
    ca = MicrobialCA(p)
    s0 = ca.sustrato.copy()
    for _ in range(20):
        ca.step(n0=3, prob_div=0.5)
    assert (ca.sustrato >= 0).all()
    assert ca.sustrato.mean() <= s0.mean()


def test_prob_div_cero_no_produce_cambio_neto():
    p = parametros_pequenos()
    ca = MicrobialCA(p)
    ca.grid[:] = 0
    ca.grid[10, 10] = 2
    antes = ca.grid.copy()
    ca.step(n0=3, prob_div=0.0)
    assert np.array_equal(ca.grid, antes)


# ---------------------------------------------------------------------------
# Test de equivalencia vectorizado vs referencia
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n0", [2, 3, 4])
@pytest.mark.parametrize("prob_div", [0.125, 0.25, 0.5])
@pytest.mark.parametrize("semilla", [7, 42, 99])
def test_step_equivale_a_la_referencia(n0: int, prob_div: float, semilla: int):
    p = parametros_pequenos()
    p.semilla = semilla
    ca_loop = MicrobialCA(p)
    ca_vec = MicrobialCA(p)
    for _ in range(30):
        step_referencia(ca_loop, n0, prob_div)
        ca_vec.step(n0, prob_div)
    assert np.array_equal(ca_loop.grid, ca_vec.grid)
    assert np.array_equal(ca_loop.sustrato, ca_vec.sustrato)