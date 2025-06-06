      subroutine refrmt (del, idip, idipdr, islip, phis, xlam)
c
c  reformat dip, strike, and rake angles to integer values and convert strike to down-dip direction
c
      real              del                             
c							! (input) fault dip angle in degrees
      integer           idip                            
c							! (output) fault dip angle in degrees
      integer           idipdr                          
c							! (output) dip direction in degrees
      integer           islip                           
c							! (output) rake in degrees
      real              phis                            
c							! (input) fault strike angle in degrees
      real              xlam                            
c							! (input) fault rake angle in degrees
c
      integer           istrk                           
c							! strike of best fit
      real              da                              
c							! dip angle of new plane
      real              dd                              
c							! dip direction angle of new plane
      real              sa                              
c							! slip angle of new plane
c
      idip = ifix(del)
      istrk = ifix(phis)
      islip = ifix(xlam)
      if (idip .gt. 90) then
        idip = 180 - idip
        istrk = istrk + 180
        islip = -islip
      else if (idip .lt. 0) then
        idip = -idip
        istrk = istrk + 180
        islip = islip + 180
      end if
      idipdr = mod(istrk + 90, 360)
      if (idipdr .lt. 0) idipdr = idipdr + 360
      islip = mod(islip, 360)
      if (islip .gt. 180) islip = islip - 360
      if (islip .lt. -180) islip = islip + 360
c
c replace each plane with idip = 0 by its auxilliary plane
c
      if (idip .eq. 0) then
        call auxpln (float(idipdr), float(idip), float(islip),
     & dd, da, sa)
        idipdr = ifix(dd)
        idip  = ifix(da)
        islip = ifix(sa)
      end if
c
c for cases where plane is vertical and dip direction .ge. 180, flip representation to one with dip direction .le. 180
c
      if (idip .eq. 90 .and. idipdr .ge. 180.) then
        islip = -islip
        idipdr = idipdr - 180.
      end if
c
      return
      end
