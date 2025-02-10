/**
 * @file
 */
#pragma once

/**
 * @brief Assigns a GPU device to the current process based on its MPI rank.
 *
 * This function determines the device assignment for a process in a distributed 
 * MPI environment. It ensures that processes running on the same node are grouped 
 * together and assigned unique device IDs.
 *
 * @return The rank of the process within its assigned node communicator (myrank).
 *         If MPI is not enabled, the function returns 0.
 */
int assignDeviceToProcess();
