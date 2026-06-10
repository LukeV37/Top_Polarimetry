c
c     Custom: require pT of the hadronic top (sum of t > b j j decay
c     products, particles 3,4,5) to be greater than __TOP_PT_CUT__ GeV
c
      ptemp(1) = P(1,3)+P(1,4)+P(1,5)
      ptemp(2) = P(2,3)+P(2,4)+P(2,5)
      if (dsqrt(ptemp(1)**2+ptemp(2)**2) .lt. __TOP_PT_CUT__d0) then
         passcuts=.false.
         return
      endif
c
