Cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
      Subroutine plssymb(x,y,itsymb,ht)
C
C     Modified from Bruce Chuchel's pltsymb for the 
C     postscript and simple plot systems on Musette
C
C     Plots a centered symbol of various kinds.
C
C                  PLTSYMB     		PLSSYMB
C                  old symbol           new symbol
C      isymb = 0 = none			box
C      isymb = 1 = plus			circle
C      isymb = 2 = box			triangle (apex up)
C      isymb = 3 = minute strike	plus
C      isymb = 4 = second strike  	X
C      isymb = 5 = X			diamond
C      isymb = 6 = circle 		triangle (down)
C      isymb = 7 = edge tic		pentagon
C      isymb = 8 = edge tic		hexagon
C      isymb = 9 = diamond		octagon
C      isymb = 10 = triangle (apex up)	X and diamond
C      isymb = 11 = triangle (apex down)X and plus
C      isymb = 12 = pentagon		box and X
C      isymb = 13 = hexagon		minute strike
C      isymb = 14 = octagon		second strike	
C              15   			edge tic (old 7)
C              16                       edge tic (old 8)
C      isymb = 19 = Point		Point
C      isymb = 20 -> 32 map back to 1 - 13
C
c      Real xcir(50),ycir(50)
      Real xcir(37),ycir(37)
      Integer nsymb(33)
C
      Character xyzprj*1
      Common /proj/xyzprj
      Common /inchbounds/xltmp,xrtmp,ybtmp,yttmp
C
C     remap old symbol numbers
C
      Data nsymb/2,6,10,1,5,9,11,12,13,14,5,5,5,3,4,7,8,3,3,19,
     &           2,6,10,1,5,9,11,12,13,14,9,1,2/
C
      Data xcir/1.000000,0.9848077,0.9396926,0.8660254,0.7660444,
     & 0.6427876,0.5000000,0.3420202,0.1736482,-4.3711388E-08,
     & -0.1736482,-0.3420201,-0.5000001,-0.6427876,-0.7660444,
     & -0.8660255,-0.9396926,-0.9848077,-1.000000,-0.9848077,
     & -0.9396926,-0.8660253,-0.7660445,-0.6427875,-0.4999999,
     & -0.3420201,-0.1736481,1.1924881E-08,0.1736481,0.3420201,
     & 0.5000004,0.6427875,0.7660443,0.8660256,0.9396926,
     & 0.9848078,1.000000/
C
      Data ycir/0.0000000E+00,0.1736482,0.3420201,0.5000000,0.6427876,
     & 0.7660444,0.8660254,0.9396926,0.9848077,1.000000,0.9848077,
     & 0.9396926,0.8660254,0.7660444,0.6427876,0.4999999,0.3420202,
     & 0.1736483,-8.7422777E-08,-0.1736482,-0.3420202,-0.5000002,
     & -0.6427876,-0.7660446,-0.8660254,-0.9396926,-0.9848078,
     & -1.000000,-0.9848077,-0.9396926,-0.8660252,-0.7660445,
     & -0.6427878,-0.4999998,-0.3420204,-0.1736480,1.7484555E-07/
C
C
C  sqrt3 = sqrt(3.0)
      Data sqrt3/1.7320508/
C
      iclip = 0
C
C - Larry Baker symbol calls.
C
C   remap symbol numbers
C
      isymb=nsymb(itsymb+1)
C
    5 continue
      If (itsymb.GE.20.AND.itsymb.LE.32) Then
C        Call symbol(x,y,ht,isymb-20,0.,-1)
         isymb=itsymb-19
         end if
C
C - Mariano/Simpson symbol call.s
C
c     Else
         ht2=0.5*ht
         x1=x-ht2
         x2=x+ht2
         y1=y-ht2
         y2=y+ht2
C
c         x1=amax1(x1,xltmp)
c         x2=amin1(x2,xrtmp)
c         y1=amax1(y1,ybtmp)
c         y2=amin1(y2,yttmp)
C
C - Plot plus
C
         If (isymb.EQ.1) Then
            Call plot(x1,y,3)
            Call plot(x2,y,2)
            Call plot(x,y1,3)
            Call plot(x,y2,2)
C
C - Plot box
C
         Else If (isymb.EQ.2) Then
            Call plot(x1,y1,3)
            Call plot(x2,y1,2)
            Call plot(x2,y2,2)
            Call plot(x1,y2,2)
            Call plot(x1,y1,2)
C
C - Plot minute strike
C
         Else If (isymb.EQ.3) Then
            y1=y-sqrt3*ht2/4
            Call plot(x-ht2/4,y1,3)
            y2=y+sqrt3*ht2/4
            Call plot(x+ht2/4,y2,2)
C
C - Plot second strike
C
         Else If (isymb.EQ.4) Then
            y1=y-sqrt3*ht2/4
            Call plot(x-ht2/4,y1,3)
            y2=y+sqrt3*ht2/4
            Call plot(x+ht2/4,y2,2)
            x=x+.02
            y3=y-sqrt3*ht2/4
            Call plot(x-ht2/4,y3,3)
            y4=y+sqrt3*ht2/4
            Call plot(x+ht2/4,y4,2)
C
C - Plot X
C
         Else If (isymb.EQ.5) Then
            Call plot(x1,y2,3)
            Call plot(x2,y1,2)
            Call plot(x1,y1,3)
            Call plot(x2,y2,2)
C
C         special code for double symbols
C
            if (itsymb.eq.10) then
              isymb=9
              goto 5
            else if (itsymb.eq.11) then
              isymb=1
              goto 5
            else if (itsymb.eq.12) then
              isymb=2
              goto 5
            end if
C
C - Plot circle
C
         Else If (isymb.EQ.6) Then
            nsides=37
            x1=x+ht2*xcir(1)
            y1=y+ht2*ycir(1)
            Call plot(x1,y1,3)
            Do 30 i=2,nsides
            x1=x+ht2*xcir(i)
            y1=y+ht2*ycir(i)
c            x1=amax1(x1,xltmp)
c            x1=amin1(x1,xrtmp)
c            y1=amax1(y1,ybtmp)
c            y1=amin1(y1,yttmp)
            Call plot(x1,y1,2)
   30       Continue
C
C - Plot edge tic
C
         Else If (isymb.EQ.7) Then
            Call plot(x,y,3)
            Call plot(x,y+ht,2)
C
C - Plot edge tic
C
         Else If (isymb.EQ.8) Then
            Call plot(x,y,3)
            Call plot(x+ht,y,2)
C
C - Plot diamond
C
         Else If (isymb.EQ.9) Then
            Call plot(x,y2,3)
            Call plot(x2,y,2)
            Call plot(x,y1,2)
            Call plot(x1,y,2)
            Call plot(x,y2,2)
C
C - Plot triangle (apex up)
C
         Else If (isymb.EQ.10) Then
            y3=y-ht/3
            x3=x-(ht/3)*sqrt3
            x4=x+(ht/3)*sqrt3
c            y3=amax1(y3,ybtmp)
c            x3=amax1(x1,xltmp)
c            x4=amin1(x2,xrtmp)
            Call plot(x,y2,3)
            Call plot(x4,y3,2)
            Call plot(x3,y3,2)
            Call plot(x,y2,2)
C
C  - Plot triangle (apex down)
C
         Else If (isymb.EQ.11) Then
            y3=y-ht/3
            x3=x-(ht/3)*sqrt3
            x4=x+(ht/3)*sqrt3
c            y3=amax1(y3,ybtmp)
c            x3=amax1(x1,xltmp)
c            x4=amin1(x2,xrtmp)
            Call plot(x4,y2,3)
            Call plot(x3,y2,2)
            Call plot(x,y3,2)
            Call plot(x4,y2,2)
C
C - Plot pentagon
C
         Else If (isymb.EQ.12) Then
C - Cos (18.0) = 0.9510565, (18.0 is in degrees).
            radius=ht2/0.9510565
            Call plotngon(x,y,5,radius,18.0,iclip,xltmp,xrtmp,
     & ybtmp,yttmp,0)
C
C - Plot hexagon
C
         Else If (isymb.EQ.13) Then
C - Sin (60.0) = 0.8660254,  (60.0 is in degrees).
            radius=ht2/0.8660254
            Call plotngon(x,y,6,radius,0.0,iclip,xltmp,xrtmp,
     & ybtmp,yttmp,0)
C
C - Plot octagon
C
         Else If (isymb.EQ.14) Then
C - Cos (22.5) = 0.9238795,  (22.5 is in degrees).
            radius=ht2/0.9238795
            Call plotngon(x,y,8,radius,22.5,iclip,xltmp,xrtmp,
     & ybtmp,yttmp,0)
         Else If (isymb.EQ.19) Then
            Call plot(x-0.004,y-0.004,3)
            Call plot(x+0.004,y+0.004,2)
         End If
c     End If
      
      Return
      End
