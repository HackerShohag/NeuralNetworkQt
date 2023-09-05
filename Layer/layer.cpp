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

#include <QtMath>
#include <QDebug>
#include <QRandomGenerator>

#include "Layer"

Layer::Layer()
{
//    qDebug() << Q_FUNC_INFO;
    this->numNodesIn = 0;
    this->numNodesOut = 0;
}

Layer::Layer(int numNodesIn, int numNodesOut)
{
    qDebug() << Q_FUNC_INFO;
    this->numNodesIn = numNodesIn;
    this->numNodesOut = numNodesOut;
    //    weights = new QVector<QVector<double>>(numNodesOut);
    //    biases = new QVector<double>(numNodesOut);
    InitializeRandomWeights();
}

Layer::~Layer()
{
//    qDebug() << Q_FUNC_INFO;
}

QVector<double> *Layer::CalculateOutputs(QVector<double> *inputs)
{
    qDebug() << Q_FUNC_INFO;
    QVector<double> *activationValues = new QVector<double>();
    for (int nodeOut = 0; nodeOut < numNodesOut; nodeOut++) {
        double weightedInput = biases->at(nodeOut);
        for (int nodeIn = 0; nodeIn < numNodesIn; nodeIn++) {
            weightedInput += inputs->at(nodeIn) * weights->at(nodeIn).at(nodeOut);
        }
        activationValues->insert(nodeOut, ActivationFunction(weightedInput));
    }
    return activationValues;
}

void Layer::ApplyGradients(double learnRate)
{
    qDebug() << Q_FUNC_INFO;
    for (int nodeOut = 0; nodeOut < numNodesOut; nodeOut++)
    {
        (*biases)[nodeOut] -= costGradientBiases->at(nodeOut) * learnRate;
        //biases->insert(nodeOut, biases->at(nodeOut) - costGradientBiases->at(nodeOut) * learnRate);
        for (int nodeIn = 0; nodeIn < numNodesIn; nodeIn++) {
            (*weights)[nodeIn][nodeOut] = costGradientWeights->at(nodeIn).at(nodeOut) * learnRate;
        }
    }
}

void Layer::InitializeRandomWeights()
{
    qDebug() << Q_FUNC_INFO;
    weights = new QVector<QVector<double>>(numNodesIn);
    for (int nodeIn = 0; nodeIn < numNodesIn; nodeIn++)
    {
        QVector<double> randomWights(numNodesOut);
        for (int nodeOut = 0; nodeOut < numNodesOut; nodeOut++)
        {
            double randomValue = (QRandomGenerator::global()->generateDouble() * 2 - 1) / qSqrt(numNodesIn);
            randomWights[nodeOut] = randomValue;
        }
        (*weights)[nodeIn] = randomWights;
    }
}

double Layer::ActivationFunction(double weightedInput)
{
    qDebug() << Q_FUNC_INFO;
    return 1 / (1 + qExp(weightedInput));
}

double Layer::NodeCost(double outputActivation, double expectedOutput)
{
    qDebug() << Q_FUNC_INFO;
    double error = outputActivation - expectedOutput;
    return error * error;
}
