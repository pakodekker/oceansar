def sinc_1tx_nrx(sin_angle, L, f0, num_chan, field=False, sin_squint=0):
    """ Calculate pattern for a system with 1 TX and N RX

        :param sin_angle: Sin of the angle
        :param L: Antenna length
        :param f0: Frequency
        :param num_ch: Number of receiving channels
        :sin_squint: Named parameter with sin of squint angle.
        :param field: If true, returned values are in EM field units, else
                      they are in power units
    """

    bp_tx = sinc_bp(sin_angle - sin_squint, L, f0, field)
    bp_rx = sinc_bp(sin_angle - sin_squint, L/num_chan, f0, field)

    return bp_tx*bp_rx