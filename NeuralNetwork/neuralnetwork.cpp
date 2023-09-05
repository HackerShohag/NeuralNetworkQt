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

#include <algorithm>
#include <QDebug>
#include "NeuralNetwork"

NeuralNetwork::NeuralNetwork(QVector<qint16> layerSizes)
{
    qDebug() << Q_FUNC_INFO;
    layers = new QVector<Layer>(layerSizes.length() - 1);
    for (int i = 0; i < layerSizes.length() - 1; i ++) {
        layers->insert(i, Layer(layerSizes.at(i), layerSizes.at(i + 1)));
    }
}

NeuralNetwork::~NeuralNetwork()
{
    qDebug() << Q_FUNC_INFO;
    delete layers;
}

QVector<double> *NeuralNetwork::CalculateOutputs(QVector<double> *inputs)
{
    qDebug() << Q_FUNC_INFO;
    for (Layer layer: *layers) {
        inputs = layer.CalculateOutputs(inputs);
    }
    return inputs;
}

int NeuralNetwork::Classify(QVector<double> *inputs)
{
    qDebug() << Q_FUNC_INFO;
    QVector<double> *outputs = CalculateOutputs(inputs);
    return *std::max_element(outputs->constBegin(), outputs->constEnd());
}

double NeuralNetwork::Cost(DataPoint *dataPoint)
{
    qDebug() << Q_FUNC_INFO;
    QVector<double> *outputs = CalculateOutputs(dataPoint->inputs);
    Layer outputLayer = layers->at(layers->length() - 1);
    double cost = 0;

    for (int nodeOut = 0; nodeOut < outputs->length(); nodeOut++)
    {
        cost += outputLayer.NodeCost(outputs->at(nodeOut), dataPoint->expectedOutputs->at(nodeOut));
    }
    return cost;
}

double NeuralNetwork::Cost(QVector<DataPoint> *data)
{
    qDebug() << Q_FUNC_INFO;
    double totalCost = 0;
    for (DataPoint dataPoint: *data)
    {
        totalCost += Cost(&dataPoint);
    }
    return totalCost / data->length();
}

//void NeuralNetwork::Learn(QVector<DataPoint> *trainingData, double learnRate)
//{
//    double h = 0.0001;
//    double originalCost = Cost(trainingData);
//    for (Layer layer : *layers)
//    {
//        // Calculate the cost gradient for the current weights
//        for (int nodeIn = 0; nodeIn < layer.numNodesIn; nodeIn++) {
//            for (int nodeOut = 0; nodeOut < layer.numNodesOut; nodeOut++) {
//                layer.weights[nodeIn][nodeOut] += h;
//                double deltaCost = Cost(trainingData) - originalCost;
//                layer.weights[nodeIn][nodeOut] -= h;
//                layer.costGradientWeights[nodeIn][nodeOut] = deltaCost / h;
//            }
//        }
//        // Calculate the cost gradient for the current biases
//        for (int biasIndex = 0; biasIndex < layer.biases->length() ; biasIndex++) {
//            layer.biases[biasIndex] += h;
//            double deltaCost = Cost(trainingData) - originalCost;
//            layer.biases[biasIndex] -= h;
//            layer.costGradientBiases [biasIndex] = deltaCost / h;
//        }
//    }
//    ApplyAllGradients(learnRate); // Calls ApplyGradients()
//}
