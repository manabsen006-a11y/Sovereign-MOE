\ TESTPROB expressed in LP format -- the same model as testprob.mps
Minimize
 COST: XONE + 2 YTWO + 3 ZTHREE
Subject To
 LIM1: XONE + YTWO <= 4
 LIM2: XONE + ZTHREE >= 1
 MYEQN: - YTWO + ZTHREE = 7
Bounds
 0 <= XONE <= 4
 -1 <= YTWO <= 1
End
