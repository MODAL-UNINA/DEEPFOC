cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
C     INTEGER FUNCTION IGETVALS
C
C    VARIABLE FORMAT READ OF VARIABLE LENGTH ARRAYS
C
C      ireturn = IGETVALS (RVAL,ILEN,IUNIT)
C
C    IRETURN = THE NUMBER OF REAL NUMBERS READ INTO THE RVAL ARRAY
C            = -1 IF THE END OF THE FILE WAS ENCOUNTERED (NO VALUES READ)
C
C    RVAL = ARRAY TO RECEIVE THE VALUES
C    ILEN = SIZE OF THE RVAL ARRAY
C    IUNIT= FORTRAN UNIT TO READ VALUES FROM
C
C     CALLS: INEXTVAL
C-
      Integer Function igetvals(rval,ilen,iunit)
      Dimension rval(ilen)
      Character*100 inbuf
      iread=0
      ierr=1
   10 Continue
      If ((iread.LE.ilen).AND.(ierr.NE.-1)) Then
         Read (iunit,100,End =30)inbuf
  100 Format (a100)
      ipos=1
      ierr=1
   20 Continue
      If ((iread.LT.ilen).AND.(ierr.EQ.1).AND.(ipos.LE.100)) Then
         iread=iread+1
         inex=inextval(inbuf(ipos:100),rval(iread),ierr)
         ipos=ipos+inex-1
         If (ierr.EQ.-2) Then
            iread=iread-1
            ierr=0
            End If
         Go To 20
         End If
      Go To 10
      End If
C
C     FALLS OUT HERE WHEN ARRAY IS FULL, OR TRANSLATION ERROR OCCURS
C
      If (ierr.EQ.-1)iread=iread-1
      igetvals=iread
      Return
C
C     HIT END OF FILE
C
   30 If (iread.EQ.0) Then
         igetvals=-1
         Else
         igetvals=iread
         End If
      End
