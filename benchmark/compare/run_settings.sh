#!/bin/bash
# Solver settings on the instances that fit: run from the repository root.
#   bash benchmark/compare/run_settings.sh <chain>      chain = inv128 | inv64dual | hydro4 | hydro8
# Each chain solves its files under the settings below (best of 3; one run above 60 s).
R="julia --project=benchmark/compare benchmark/compare/run_solves.jl"
case "$1" in
inv128)
  for s in ipm ipm_nocross primal_simplex nopresolve; do
    $R setting=$s inventory_T128_new_primal inventory_T128_legacy_primal inventory_T128_legacy_primal_sae
  done ;;
inv64dual)
  for s in ipm ipm_nocross primal_simplex nopresolve; do
    $R setting=$s inventory_T64_new_dual inventory_T64_legacy_dual
  done ;;
hydro4)
  for s in ipm ipm_nocross primal_simplex nopresolve; do
    $R setting=$s hydro_T4_new_primal hydro_T4_legacy_primal_sae hydro_T4_legacy_primal hydro_T4_new_dual hydro_T4_legacy_dual
  done ;;
hydro8)
  for s in ipm ipm_nocross; do
    $R reps=1 setting=$s hydro_T8_new_primal hydro_T8_legacy_primal_sae hydro_T8_new_dual hydro_T8_legacy_dual
  done ;;
esac
