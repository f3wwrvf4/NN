
#include "NN_net.h"
#include <iostream>
#include <iomanip>
#include <fstream>
#include <vector>

namespace NN
{
template <class T>
void Layer<T>::save(std::ostream& ost) const
{
  ost << weight;
}

template <class T>
void Layer<T>::load(std::istream& ist)
{
  ist >> weight;
}

Network::Network(int layer_num_, const int* node_num_, int batch_num_):
  layer_num(layer_num_),
  batch_num(batch_num_)
{
  node_num = new int[layer_num];
  for (int i = 0; i < layer_num; ++i) {
    node_num[i] = node_num_[i];
  }

  const int layers_len = layer_num - 1;
  const int last_layer = layer_num - 1;

  layers = new Layer<Logistic>*[layers_len];

  NN::Layer<Logistic>* prev = 0;
  for (int i = 0; i < layers_len; ++i) {
    const bool isLast = i == (layers_len - 1);
    const int node1 = node_num[i] + 1;
    const int node2 = isLast ? node_num[i + 1] : node_num[i + 1] + 1;
    layers[i] = new NN::Layer<NN::Logistic>(batch_num, node1, node2);
    layers[i]->prev = prev;
    if (prev) prev->next = layers[i];
    prev = layers[i];
  }
}

Network::~Network()
{
  const int layers_len = layer_num - 1;
  for (int i = 0; i < layers_len; ++i) {
    delete layers[i];
  }
  delete[] layers;

  delete[] node_num;
}

void Network::train(const Matrix& input, const Matrix& output)
{
  const int layers_len = layer_num - 1;
  const int last_layer = layer_num - 1;

  const int Node1 = node_num[0] + 1;
  const int NodeL = node_num[last_layer];

//  NN::Matrix input(batch_num, Node1);
  NN::Matrix delta_o(batch_num, NodeL);
  NN::Layer<Logistic>* layer;

  float error = 0;
  int iTrain = 0;
  while (true) {
    ++iTrain;
    error = 0;

    layer = layers[0];
    const NN::Matrix* in = &input;
    for (int i = 0; i < layers_len; ++i) {
      in = layer->forward(in);
      layer = layer->next;
    }

    // error
    const Matrix& out = layers[layers_len - 1]->out;
    for (int i = 0; i < batch_num; ++i) {
      for (int j = 0; j < NodeL; ++j) {
        const float err = NN::Square(output(i, j) - out(i, j));
        error += err;
        delta_o(i, j) = out(i, j) - output(i, j);
      }
    }

    if (error < 0.1f) {
      for (int i = 0; i < batch_num; ++i) {
        for (int j = 0; j < NodeL; ++j) {
          float t = output(i, j);
          float o = out(i, j);
//          std::cout << t << " - " << o << std::endl;
        }
      }
//      std::cout << iTrain << ":" << error << std::endl;
//      std::cout << "› " << std::setw(6) << iTrain << ":" << std::setw(8) << std::left << error << std::endl;
      break;
    }else if((iTrain%2000)==0){

//      std::cout << "~ "<< std::setw(6) << iTrain << ":" << std::setw(8) << std::left << error << std::endl;
//      std::cout << input;
//      std::cout << output;
//      std::cout << out;

      break;
    }


    // back
    layer = layers[layers_len - 1];
    const Matrix* delta = &delta_o;
    while (layer) {
      delta = layer->back(delta);
      layer = layer->prev;
    }
  }
}

const Matrix& Network::eval(const Matrix& input) const
{
  _ASSERT(input.row() == 1);
  _ASSERT(input.col() == node_num[0] + 1);
//  input(0, node_num[0]) = 1.0f;

  const NN::Matrix* in = &input;
  for (int i = 0; i < layer_num - 1; ++i) {
    in = layers[i]->eval(in);
  }
  return layers[layer_num-2]->out_vec;
}


void Network::save(const char* fpath) const
{
  std::ofstream ofs(fpath);

  // input
  ofs << layer_num << " ";
  for (int i = 0; i < layer_num; ++i) {
    int sz = node_num[i];
    ofs << sz << " ";
  }

  for (int i = 0; i < layer_num - 1; ++i) {
    layers[i]->save(ofs);
  }

  ofs << std::endl;
}

Network* Network::create(const char* fpath)
{
#if 0
  int layer_num;
  int* node_num;
  int batch_num;

  std::ifstream ifs(fpath);
  ifs >> layer_num;
  node_num = new int[layer_num];
  for (int i = 0; i < layer_num; ++i) {
    ifs >> node_num[i];
  }

  Network* net = new Network(layer_num, node_num, batch_num);
  for (int i = 0; i < layer_num - 1; ++i) {
    net->layers[i]->load(ifs);
  }
  return net;
#else
  return 0;
#endif
}

void Network::load(const char* fpath)
{
  int ival;

  std::ifstream ifs(fpath);
  ifs >> ival;
  if (ival != layer_num) return;

  for (int i = 0; i < layer_num; ++i) {
    int sz = node_num[i];
    ifs >> ival;
    if (ival != sz) return;
  }

  for (int i = 0; i < layer_num - 1; ++i) {
    layers[i]->load(ifs);
  }
}

#if 0
Network::Network(int* L, int l_num) :
  L_NUM(l_num),
  W_NUM(l_num - 1),
  I_NUM(l_num),
  L_LAST(l_num - 1),
  D_NUM(l_num - 1),
  D_LAST((l_num - 1) - 1)
{

  weights = new Matrix*[W_NUM];
  inputs = new Vector*[I_NUM];
  backs = new Vector*[I_NUM];
  deltas = new Vector*[D_NUM];
  deltas_t = new Matrix*[D_NUM];

  for (int i = 0; i < I_NUM; ++i) {
    int sz;
    if (i == L_LAST) {
      sz = L[i];
    } else {
      sz = L[i] + 1; // weight‚ðÜ‚èž‚Þ
    }
    inputs[i] = new Vector(sz);
    backs[i] = new Vector(sz);
  }

  for (int i = 0; i < W_NUM; ++i) {
    const int szcol = inputs[i]->size();
    const int szrow = inputs[i + 1]->size();

    weights[i] = new Matrix(szrow, szcol);

    float v = 1.0f;
    for (int r = 0; r < szcol; ++r) {
      for (int c = 0; c < szrow; ++c) {
        (*weights[i])(c, r) = (rand() / (float)RAND_MAX) - 0.5f;
      }
    }
  }

  for (int i = 0; i < D_NUM; ++i) {
    deltas[i] = new Vector(inputs[i + 1]->size());
    deltas_t[i] = new Matrix(1, inputs[i + 1]->size());
  }
}
Network::~Network()
{
  for (int i = 0; i < I_NUM; ++i) {
    delete inputs[i];
  }
  delete[] inputs;

  for (int i = 0; i < W_NUM; ++i) {
    delete weights[i];
  }
  delete[] weights;

  for (int i = 0; i < D_NUM; ++i) {
    delete deltas[i];
  }
  delete[] deltas;
}

void Network::train(const TrainData* td_tbl, int num)
{
  const int DATA_COUNT = num;
  const int TRAIN_COUNT = 10000000;

  for (int iTrainCount = 0; iTrainCount < TRAIN_COUNT; ++iTrainCount) {
    float error = 0;
    for (int iDataCount = 0; iDataCount < DATA_COUNT; ++iDataCount) {
      const TrainData& data = td_tbl[iDataCount];
      {
        Vector& y = *inputs[0];
        for (int i = 0; i < y.size() - 1; ++i) {
          y(i) = data.input[i];
        }
      }

      // forwad
      feedForward();

      // Œë·‚Ì•]‰¿
      {
        const Vector& out = *inputs[L_LAST];
        for (int i = 0; i < out.size(); ++i) {
          error += ((data.output[i] - out(i))*(data.output[i] - out(i)));
        }
      }

      // back
      {
        backPropagate(data);
      }
    }

    if ((iTrainCount % 100) == 0) {
      std::cout << "Train Count:" << iTrainCount
        << "  error:" << error << std::endl;
    }

    if (error < 0.01f) {
      std::cout << "err:" << error << std::endl;
      break;
    }
  }
}

void Network::feedForward()
{
  // ‡“`”d
  for (int iL = 0; iL < L_NUM - 1; ++iL) {
    Vector& y = (*inputs[iL]);
    Matrix& w = (*weights[iL]);
    Vector& o = (*inputs[iL + 1]);
    if (iL < L_LAST) {
      const int last = y.size() - 1;
      y(last) = 1.0f;  // weight‚ð“ü—Í‚Æ‚µ‚Ä•]‰¿‚·‚é
    }
    NN::Mul(y, w, o);
    for (int i = 0; i < o.size(); ++i) {
      o(i) = sigmoid(o(i));
    }
  }
}

void Network::backPropagate(const TrainData& data)
{
  {
    // o—Í‘w
    {
      const Vector& out = *inputs[L_LAST];
      Vector& delta = *deltas[D_LAST];
      for (int i = 0; i < out.size(); ++i) {
        delta(i) = (out(i) - data.output[i]) * (1.0f - out(i)) * out(i);
      }
    }

    // ‰B‚ê‘w
    for (int i = W_NUM - 1; 0 < i; --i) {
      const int layer = i;
      const Vector& hidden = *inputs[layer];
      const Matrix& weight = *weights[layer];
      const Vector& prev_delta = *deltas[layer];
      Matrix& prev_delta_t = *deltas_t[layer];
      Vector& hid_back = *deltas[layer - 1];
      prev_delta.t(prev_delta_t);
      Mul(weight, prev_delta_t, hid_back);
      for (int i = 0; i < hid_back.size(); ++i) {
        hid_back(i) = hid_back(i) * (1.0f - hidden(i)) * hidden(i);
      }
    }

    // d‚Ý‚ðC³
    const float eps = 0.1f;
    for (int i = 0; i < W_NUM; ++i) {
      Matrix& weigh = *weights[i];
      const Vector& back = *deltas[i];
      const Vector& inp = *inputs[i];

      for (int iRow = 0; iRow < weigh.row(); ++iRow) {
        for (int iCol = 0; iCol < weigh.col(); ++iCol) {
          float b = back(iRow);
          float in = inp(iCol);
          weigh(iRow, iCol) -= eps*in*b;
        }
      }
    }
  }
}

std::vector<float> Network::eval(std::vector<float> input)
{
  Vector& y = *inputs[0];
  for (int i = 0; i < y.size() - 1; ++i) {
    y(i) = input[i];
  }
  feedForward();
  return (*inputs[L_LAST]).vec();
}

void Network::save(const char* fpath)
{
  std::ofstream ofs(fpath);

  // input
  ofs << I_NUM << std::endl;
  for (int i = 0; i < I_NUM; ++i) {
    int sz = inputs[i]->size();
    if (i != (I_NUM - 1)) {
      sz -= 1;
    }
    ofs << sz << " ";
  }
  ofs << std::endl;

  // weight
  ofs << W_NUM << std::endl;
  for (int i = 0; i < W_NUM; ++i) {
    ofs << (*weights[i]);
  }
  ofs << std::endl;
}

Network* Network::CreateFromFile(const char* fpath)
{
  std::ifstream ifs(fpath);
  int i_num;
  ifs >> i_num;
  int* ary = new int[i_num];
  for (int i = 0; i < i_num; ++i) {
    ifs >> ary[i];
  }
  Network* net = new Network(ary, i_num);
  delete[] ary;

  int w_num;
  ifs >> w_num;
  for (int i = 0; i < w_num; ++i) {
    ifs >> *net->weights[i];
  }
  return net;
}
#endif
}
