/*
 * Copyright (C) 2023  Abdullah AL Shohag
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; version 3.
 *
 * raven.downloader is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

#ifndef LAYER_H
#define LAYER_H

#include <QVector>

class Layer
{
public:
    Layer();
    Layer(int numNodesIn, int numNodesOut);
    ~Layer();
    QVector<double> *CalculateOutputs(QVector<double> *inputs);
    void ApplyGradients(double learnRate);
    void InitializeRandomWeights();
    double ActivationFunction(double weightedInput);
    double NodeCost(double outputActivation, double expectedOutput);

    // getters
    int getNumNodesOut() const;
    int getNumNodesIn() const;
    QVector<QVector<double> > *getCostGradientWeights() const;
    QVector<double> *getCostGradientBiases() const;
    QVector<QVector<double> > *getWeights() const;
    QVector<double> *getBiases() const;

private:
    int numNodesIn, numNodesOut;

    QVector<QVector<double>> *costGradientWeights;
    QVector<double> *costGradientBiases;

    QVector<QVector<double>> *weights;
    QVector<double> *biases;
};

#endif // LAYER_H
