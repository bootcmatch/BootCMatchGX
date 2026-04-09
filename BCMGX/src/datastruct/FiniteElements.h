#pragma once

struct Node
{
  double x, y;
  int gn;   // indice globale nodo
};

struct Element
{
  int n[3]; // nodi di ogni elemento triangolare (locali)
  int gn[3];     // globali
};

struct Triplet
{
  int row;
  int col;
  double val;
};