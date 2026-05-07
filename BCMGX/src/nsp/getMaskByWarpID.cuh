// Function to get the partial mask of a miniwarp of size m_size, given its group id m_id
__device__ __inline__ unsigned int getMaskByWarpID( const unsigned int &m_size,
                                                    const unsigned int &m_id ) {

   if (m_size == 32) {
       return MASKFULL;
   }

   unsigned int m = (1 << (m_size)) - 1;

   return (m << (m_size * m_id));
}
