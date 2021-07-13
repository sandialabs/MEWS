      PROGRAM WTHFMT
C
C     THIS PROGRAM READS A PACKED BINARY DOE-2 WEATHER FILE AND
C     CREATES A FORMATTED WEATHER FILE AS OUTPUT.  THE INPUT
C     FILE IS TAPE10, THE OUTPUT FILE IS TAPE11.
C
      DIMENSION CLN(12),GT(12),MDAYS(12),IDAT30(1536),
     _          IWDID(5),IWTH(14)
      DIMENSION XMASK(16,2), CALC(16)
      INTEGER IDUM
C
      DATA MDAYS / 31,28,31,30,31,30,31,31,30,31,30,31 /
      DATA IWDID /5*4H    /
      DATA XMASK / -99., -99., 15., 0., 0., 0., 0., 0., .02, -30., 0.,
     1             .0, .0, .0, .0, 10., 
     2              1., 1., .1, 1., 1., 1., 1., .0001, .001, .5,
     3             1., 1., 1., 1., 0., 0. /
C
      OPEN (UNIT=11,FILE='WEATHER.FMT',blocksize=710000)
      OPEN (UNIT=10,FILE='WEATHER.BIN',FORM='UNFORMATTED',
     $      recordtype='variable',recl=6200,blocksize=148992)
      REWIND 10
      DO 100 IM1=1,12
      PRINT *, "READ 1"
      READ (10) (IWDID(I),I=1,5),IWYR,WLAT,WLONG,IWTZN,LRECX,NUMDAY,
     _          CLN(IM1),GT(IM1),IWSOL
      READ (10) IDUM
  100 CONTINUE
      REWIND 10
      LRECX = 0
      WRITE (11,9001) (IWDID(I),I=1,5),IWYR,WLAT,WLONG,IWTZN,IWSOL
      WRITE (11,9002) (CLN(I),I=1,12)
      WRITE (11,9003) (GT(I),I=1,12)
 9001 FORMAT(5A4,I5,2F8.2,2I5)
 9002 FORMAT(12F6.2)
 9003 FORMAT(12F6.1)
      DO 1000 IM2=1,12
      IDE = MDAYS(IM2)
      DO 1000 ID=1,IDE
      DO 1000 IH=1,24
  105 IRECX = IM2*2 + (ID-1)/16 - 1
      IDX = MOD(ID-1,16) + 1
      IF (IRECX-LRECX) 200,400,300
  200 IDIF = LRECX - IRECX + 1
      DO 220 I=1,IDIF
      BACKSPACE 10
  220 CONTINUE
  300 READ (10) IWDID,IWYR,WLAT,WLONG,IWTZN,LRECX,NUMDAY,CLRNES,
     _          TGRND,IDUM,IDAT30
      GO TO 105
  400 CONTINUE
      IP1 = 96*(IDX-1) + 4*IH - 3
      IWTH(3) = IDAT30(IP1)/65536
      IWTH(1) = MOD(IDAT30(IP1),65536)/256
      IWTH(2) = MOD(IDAT30(IP1),256)
      IWTH(11) = IDAT30(IP1+1)/1048576
      IWTH(12) = MOD(IDAT30(IP1+1),1048576)/1024
      IWTH(4) = MOD(IDAT30(IP1+1),1024)/64
      IWTH(5) = MOD(IDAT30(IP1+1),64)/32
      IWTH(6) = MOD(IDAT30(IP1+1),32)/16
      IWTH(7) = MOD(IDAT30(IP1+1),16)
      IWTH(8) = IDAT30(IP1+2)/128
      IWTH(9) = MOD(IDAT30(IP1+2),128)
      IWTH(10) = IDAT30(IP1+3)/2048
      IWTH(13) = MOD(IDAT30(IP1+3),2048)/128
      IWTH(14) = MOD(IDAT30(IP1+3),128)
      DO 500 I=1,14
      CALC(I) = FLOAT(IWTH(I))*XMASK(I,2) + XMASK(I,1)
  500 CONTINUE    
      ISNOW = INT(CALC(5) + .01)
      IRAIN = INT(CALC(6) + .01)
      IWNDDR = INT(CALC(7) + .01)
      ICLDTY = INT(CALC(13) + .01)
C              
C              IM2            MOMTH              (1-12)
C              ID             DAY OF MONTH
C              IH             HOUR OF DAY
C              CALC(1)        WET BULB TEMP      (DEG F)
C              CALC(2)        DRY BULB TEMP      (DEG F)
C              CALC(3)        PRESSURE           (INCHES OF HG)
C              CALC(4)        CLOUD AMOUNT       (0 - 10)
C              ISNOW          SNOW FLAG          (1 = SNOWFALL)
C              IRAIN          RAIN FLAG          (1 = RAINFALL)
C              IWNDDR         WIND DIRECTION     (0 - 15; 0=N, 1=NNE, ETC)
C              CALC(8)        HUMIDITY RATIO     (LB H2O/LB AIR)
C              CALC(9)        DENSITY OF AIR     (LB/CU FT)
C              CALC(10)       SPECIFIC ENTHALPY  (BTU/LB)
C              CALC(11)       TOTAL HOR. SOLAR   (BTU/HR-SQFT)
C              CALC(12)       DIR. NORMAL SOLAR  (BTU/HR-SQFT)
C              ICLDTY         CLOUD TYPE         (0 - 2)
C              CALC(14)       WIND SPEED         KNOTS
C
  900 WRITE (11,9005) IM2, ID, IH, CALC(1), CALC(2), CALC(3), CALC(4),
     1                ISNOW, IRAIN,IWNDDR, CALC(8), CALC(9), CALC(10),
     2                CALC(11), CALC(12), ICLDTY, CALC(14)
 9005 FORMAT(3I2,2F5.0,F6.1,F5.0,2I3,I4,F7.4,F6.3,F6.1,2F7.1,I3,F5.0)
 1000 CONTINUE
      ENDFILE 11
      END
